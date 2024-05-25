# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autocast (https://arxiv.org/abs/2206.15474) implementation
# from https://github.com/andyzoujm/autocast by Andy Zou and Tristan Xiao and Ryan Jia and Joe Kwon and Mantas Mazeika and Richard Li and Dawn Song and Jacob Steinhardt and Owain Evans and Dan Hendrycks
####################################################################################

import torch
import pickle
from torch.utils import tensorboard
import transformers
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)
import torch.distributed as dist

from src.argparser import arg_parser
import src.slurm
import src.util
import src.evaluation
import src.data_loader
import src.model_multihead


def train(
    model,
    optimizer,
    scheduler,
    step,
    train_dataset,
    eval_dataset,
    opt,
    tokenizer,
    collator,
    best_dev_em,
    checkpoint_path,
    logger,
):
    """
    Train the model
    """

    # set up tensorboard
    if opt.is_main:
        tb_logger = tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
    else:
        tb_logger = None

    # set up DDP dataloader
    if opt.is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=np.clip(os.cpu_count(), a_min=4, a_max=8),
        collate_fn=collator,
        pin_memory=True,
    )

    if opt.loss_reweight:
        freq_per_question_type = {
            k: len(v) for k, v in train_dataset.data_by_class.items()
        }
        weight_per_question_type = {
            k: 1.0 / v for k, v in freq_per_question_type.items()
        }
        # let t/f type loss be 1.0, adjust the other two types accordingly
        weight_per_question_type["mc"] = (
            weight_per_question_type["mc"] / weight_per_question_type["t/f"]
        )
        weight_per_question_type["num"] = (
            weight_per_question_type["num"] / weight_per_question_type["t/f"]
        )
        weight_per_question_type["t/f"] = 1.0
    else:
        weight_per_question_type = {"t/f": 1.0, "mc": 1.0, "num": 1.0}

    logger.info("Weight per question type: {}".format(weight_per_question_type))

    # go training
    for epoch in range(opt.epochs):
        if opt.is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
        model.train()

        epoch += 1
        epoch_loss, epoch_loss_tf, epoch_loss_mc, epoch_loss_num, total_dps = (
            [],
            [],
            [],
            [],
            0,
        )
        epoch_loss_aln_tf, epoch_loss_aln_mc, epoch_loss_aln_num = [], [], []
        epoch_acc_tf, epoch_acc_mc, epoch_acc_num = [], [], []

        for i, batch in enumerate(train_dataloader):
            step += 1
            (
                idx,
                ids,
                labels,
                indices,
                lengths,
                context_ids,
                context_mask,
                targets_raw,
                human_forecasts,
                forecast_time_orders,
            ) = batch
            total_dps += len(ids)
            # dataloader returns:
            # index: dataset-specific integer index, tensor of integers                         [B]
            # ids: original question IDs, list of strings                                       [B]
            # labels: tokenized question answers, tensor of integers                            [B, L=10]
            # indices: question types, tensor of integers, t/f + mc + num                       [3, N]
            # lengths: count of questions of each type, tensor of integers, t/f + mc + num      [3]
            # passage_ids: tokenized passage IDs, tensor of integers                            [B, N, D=512]
            # passage_masks: passage masks, tensor of booleans                                  [B, N, D=512]
            # targets_raw: raw answers, list of strings                                         [B]
            # human_forecasts: human forecasts for the news articles, tensor of floats          [B, N]
            # forecast_time_orders: temporal order of the news articles, tensor of integers     [B, N]
            # note: the prepended news articles context_ids already contain the question text

            if (
                i % max(len(train_dataloader) // 10, 1) == 0
                or i == len(train_dataloader) - 1
                or i == 0
            ):
                logger.info(
                    "Rank {:d}. Epoch: {:03d} / {:03d}. Training progress: {:04d} / {:04d} iterations, {:04d} / {:04d} instances. LR: {:.4E}.".format(
                        opt.local_rank,
                        epoch,
                        opt.epochs,
                        i + 1,
                        len(train_dataloader),
                        total_dps,
                        len(train_dataloader.sampler),
                        optimizer.param_groups[0]["lr"],
                    )
                )

            train_loss, loss_per_cat, loss_aln_per_cat, acc_per_cat = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                indices=indices.cuda(),
                lengths=lengths.cuda(),
                labels=labels.cuda(),
                human_forecasts=human_forecasts.cuda(),
                forecast_time_orders=forecast_time_orders.cuda(),
                loss_reweight=weight_per_question_type if opt.loss_reweight else None,
            )[
                2:
            ]  # original output tuple: logits, previous_outputs, loss_training, loss_per_cat, acc_per_cat

            loss_tf, loss_mc, loss_num = loss_per_cat
            loss_aln_tf, loss_aln_mc, loss_aln_num = loss_aln_per_cat
            acc_tf, acc_mc, acc_num = acc_per_cat

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)

            train_loss = src.util.average_main(train_loss, opt)
            loss_tf = src.util.average_main(loss_tf, opt)
            loss_mc = src.util.average_main(loss_mc, opt)
            loss_num = src.util.average_main(loss_num, opt)
            loss_aln_tf = src.util.average_main(loss_aln_tf, opt)
            loss_aln_mc = src.util.average_main(loss_aln_mc, opt)
            loss_aln_num = src.util.average_main(loss_aln_num, opt)
            acc_tf = src.util.average_main(acc_tf, opt)
            acc_mc = src.util.average_main(acc_mc, opt)
            acc_num = src.util.average_main(acc_num, opt)

            epoch_loss.append(train_loss.item())
            epoch_loss_tf.append(loss_tf.item())
            epoch_loss_mc.append(loss_mc.item())
            epoch_loss_num.append(loss_num.item())
            epoch_loss_aln_tf.append(loss_aln_tf.item())
            epoch_loss_aln_mc.append(loss_aln_mc.item())
            epoch_loss_aln_num.append(loss_aln_num.item())
            epoch_acc_tf.append(acc_tf.item())
            epoch_acc_mc.append(acc_mc.item())
            epoch_acc_num.append(acc_num.item())

            if tb_logger is not None:
                tb_logger.add_scalar(
                    "Training_iteration/loss_training", train_loss.item(), step
                )
                tb_logger.add_scalar("Training_iteration/loss_tf", loss_tf.item(), step)
                tb_logger.add_scalar("Training_iteration/loss_mc", loss_mc.item(), step)
                tb_logger.add_scalar(
                    "Training_iteration/loss_num", loss_num.item(), step
                )
                tb_logger.add_scalar(
                    "Training_iteration/loss_aln_tf", loss_aln_tf.item(), step
                )
                tb_logger.add_scalar(
                    "Training_iteration/loss_aln_mc", loss_aln_mc.item(), step
                )
                tb_logger.add_scalar(
                    "Training_iteration/loss_aln_num", loss_aln_num.item(), step
                )
                tb_logger.add_scalar(
                    "Training_iteration/accuracy_tf", acc_tf.item(), step
                )
                tb_logger.add_scalar(
                    "Training_iteration/accuracy_mc", acc_mc.item(), step
                )
                tb_logger.add_scalar(
                    "Training_iteration/accuracy_num", acc_num.item(), step
                )
                tb_logger.add_scalar(
                    "Meta/Learning_rate_step", optimizer.param_groups[0]["lr"], step
                )

            # the end of each training iteration
        # the end of each epoch
        if opt.is_distributed:
            dist.barrier()

        logger.info(
            f"Rank {opt.local_rank}. Epoch {epoch} training is finished with {total_dps} datapoints. Now proceed to evaluation."
        )
        num_iter_per_epoch = len(train_dataloader)
        if tb_logger is not None:
            tb_logger.add_scalar(
                "Training_epoch/loss_training", np.mean(epoch_loss), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/loss_tf", np.mean(epoch_loss_tf), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/loss_mc", np.mean(epoch_loss_mc), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/loss_num", np.mean(epoch_loss_num), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/loss_aln_tf", np.mean(epoch_loss_aln_tf), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/loss_aln_mc", np.mean(epoch_loss_aln_mc), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/loss_aln_num", np.mean(epoch_loss_aln_num), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/accuracy_tf", np.mean(epoch_acc_tf), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/accuracy_mc", np.mean(epoch_acc_mc), epoch
            )
            tb_logger.add_scalar(
                "Training_epoch/accuracy_num", np.mean(epoch_acc_num), epoch
            )

        # go evaluation
        train_em = 0.0  # skip training evaluation to save time and avoid DDP error due to timeout
        dev_em = evaluate(
            model,
            eval_dataset,
            tokenizer,
            collator,
            opt,
            epoch,
            logger,
            checkpoint_path,
            tb_logger=tb_logger,
        )
        if opt.is_main:
            if dev_em > best_dev_em:
                best_dev_em = dev_em
                src.util.save(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_dev_em,
                    opt,
                    checkpoint_path,
                    "best_dev",
                )
                logger.info(
                    "Saving best model at epoch {:d} to {}".format(
                        epoch, checkpoint_path
                    )
                )
            log = f"Epochs: {epoch} / {opt.epochs} | "
            log += f"train loss: total: {np.mean(epoch_loss):.3f}; tf: {np.mean(epoch_loss_tf): .3f}; mc: {np.mean(epoch_loss_mc): .3f}; num: {np.mean(epoch_loss_num): .3f} | "
            log += f"lr: {optimizer.param_groups[0]['lr']:.3E}"
            logger.info(log)

            log = f"Epochs: {epoch} / {opt.epochs} | "
            log += f"Train EM: {100*train_em:.2f}, Dev EM: {100*dev_em:.2f} "
            logger.info(log)
            if tb_logger is not None:
                tb_logger.add_scalar(
                    "Meta/Learning_rate_epoch", optimizer.param_groups[0]["lr"], epoch
                )
                tb_logger.add_scalar("Training_epoch/Train_EM", train_em, epoch)
                tb_logger.add_scalar("Training_epoch/Dev_EM", dev_em, epoch)

            # save checkpoint
            if epoch % opt.save_freq == 0:
                src.util.save(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_dev_em,
                    opt,
                    checkpoint_path,
                    f"epoch-{epoch}-step-{step}",
                )
                logger.info(
                    "Saving checkpoint model at epoch {:d} to {}".format(
                        epoch, checkpoint_path
                    )
                )

        if not opt.epochs and step > opt.total_steps:
            return

    logger.info("Training is done!")

    if opt.is_main:
        src.util.save(
            model,
            optimizer,
            scheduler,
            step,
            best_dev_em,
            opt,
            checkpoint_path,
            f"epoch-{epoch}",
        )
        logger.info("Saving final model to {}".format(checkpoint_path))


def evaluate(
    model,
    dataset,
    tokenizer,
    collator,
    opt,
    epoch,
    logger,
    checkpoint_path,
    mode="eval",
    tb_logger=None,
):
    """
    Evaluate the model
    """

    # set up tensorboard
    if opt.is_main and tb_logger is None:
        tb_logger = tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)

    # a set of fixed tokens for different question types
    TF_TOKENS = sum(tokenizer(["no", "yes"])["input_ids"], [])
    MC_TOKENS = sum(tokenizer([chr(i + ord("A")) for i in range(12)])["input_ids"], [])
    NUM_TOKENS = sum(tokenizer([chr(i + ord("A")) for i in range(20)])["input_ids"], [])

    # set up DDP dataloader
    if opt.is_distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size * 2,  # larger batch size in eval mode
        drop_last=False,
        num_workers=np.clip(os.cpu_count(), a_min=4, a_max=8),
        collate_fn=collator,
        pin_memory=True,
    )

    # go evaluation
    model.eval()
    total_questions = 0
    device = torch.device("cpu")
    question_ids, gt_answers = [], []
    tf_ans, mc_ans, num_char_ans, num_float_ans, global_ans = [], [], [], [], []
    tf_em, mc_em, num_char_em, num_float_em, global_em = [], [], [], [], []
    tf_logits, mc_logits, num_logits, global_logits = [], [], [], []

    # helper for numerical question evaluation
    bins_interval = np.linspace(0, 1, dataset.numerical_bins)
    bins_letter2float = {
        chr(ord("A") + i): bins_interval[i] for i in range(dataset.numerical_bins)
    }

    with torch.no_grad():
        if opt.is_distributed:
            dataloader.sampler.set_epoch(0)
        for i, batch in enumerate(dataloader):
            (
                idx,
                ids,
                labels,
                indices,
                lengths,
                context_ids,
                context_mask,
                targets_raw,
                _,
                _,
            ) = batch
            # dataloader returns:
            # index: dataset-specific integer index, tensor of integers                         [B]
            # ids: original question IDs, list of strings                                       [B]
            # labels: tokenized question answers, tensor of integers                            [B, L=10]
            # indices: question types, tensor of integers, t/f + mc + num                       [3, N]
            # lengths: count of questions of each type, tensor of integers, t/f + mc + num      [3]
            # passage_ids: tokenized passage IDs, tensor of integers                            [B, N, D=512]
            # passage_masks: passage masks, tensor of booleans                                  [B, N, D=512]
            # targets_raw: raw question answers, list of strings                                [B]
            # note: the prepended news articles context_ids already contain the question text

            total_questions += len(ids)
            if (
                i % max(len(dataloader) // 10, 1) == 0
                or i == len(dataloader) - 1
                or i == 0
            ):
                logger.info(
                    "Rank {:d}. Epoch: {:03d} / {:03d}. Evaluate progress: {:04d} / {:04d} iterations, {:04d} / {:04d} instances.".format(
                        opt.local_rank,
                        epoch,
                        opt.epochs,
                        i + 1,
                        len(dataloader),
                        total_questions,
                        len(dataloader.sampler),
                    )
                )

            func_generate = (
                model.module.generate if hasattr(model, "module") else model.generate
            )
            output_generation = func_generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=10,
                indices=indices.cuda(),
                lengths=lengths.cuda(),
                return_dict_in_generate=True,
                output_scores=True,
            )
            id_outputs, logits = (
                output_generation.sequences,
                output_generation.scores,
            )  # [B, 10] + [B, V] * 9
            output_logits = (
                torch.stack(logits).swapaxes(0, 1).detach().to(device)
            )  # [B, 9, V]

            # parse the outputs
            indices_tf = indices[0][: lengths[0]]
            indices_mc = indices[1][: lengths[1]]
            indices_num = indices[2][: lengths[2]]

            question_ids.extend(ids)
            gt_answers.extend(targets_raw)
            for k, (o, lgs) in enumerate(zip(id_outputs, output_logits)):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                if k not in indices_num:
                    gold = [str(dataset.get_example(idx[k])["answers"])]
                else:
                    gold = [str(dataset.get_example(idx[k])["answers_bin_char"])]
                flag_exact_match = src.evaluation.ems(ans, gold)
                assert flag_exact_match in [True, False]

                global_ans.append(ans)
                global_em.append(flag_exact_match)

                if k in indices_tf:
                    tf_ans.append(ans)
                    tf_em.append(flag_exact_match)
                    tf_logits.append(lgs[0, TF_TOKENS])
                    global_logits.append(lgs[0, TF_TOKENS])
                elif k in indices_mc:
                    mc_ans.append(ans)
                    mc_em.append(flag_exact_match)
                    mc_logits.append(lgs[0, MC_TOKENS])
                    global_logits.append(lgs[0, MC_TOKENS])
                elif k in indices_num:
                    num_char_ans.append(ans)
                    num_char_em.append(flag_exact_match)
                    num_logits.append(lgs[0, NUM_TOKENS])
                    global_logits.append(lgs[0, NUM_TOKENS])

                    if ans in bins_letter2float.keys():
                        assert len(ans) == 1 and "A" <= ans < chr(
                            ord("A") + dataset.numerical_bins
                        )
                        num_float_ans.append(bins_letter2float[ans])
                        num_float_em.append(
                            abs(float(targets_raw[k]) - num_float_ans[-1])
                        )
                    else:
                        num_float_ans.append(None)
                        num_float_em.append(1.0)  # max error for invalid answers

    # convert logits tensors to numpy array for pickle ops
    global_logits_ = []
    for item in global_logits:
        if isinstance(item, torch.Tensor):
            global_logits_.append(item.to(device).numpy())
        else:
            global_logits_.append(np.array(item))
    global_logits = global_logits_

    if opt.is_distributed:
        dist.barrier()
        logger.info(
            "Rank: {:d}. Inference for {:s} set at epoch {:d} is done (with {:d} instances).".format(
                opt.local_rank, mode, epoch, len(question_ids)
            )
        )
        objects = [
            tf_em,
            mc_em,
            num_char_em,
            num_float_em,
            tf_ans,
            mc_ans,
            num_char_ans,
            num_float_ans,
            global_logits,
            question_ids,
            gt_answers,
        ]
        all_objects = [None for _ in range(opt.world_size)]
        stats_before_gather = (
            "Total number of DPs before gather: {:d}. Mode: {:s}".format(
                len(tf_em) + len(mc_em) + len(num_char_em), mode
            )
        )  # diagnosis purpose

        # note: gathering into the main GPU may be time-consuming for large dataset because of global_logits
        dist.all_gather_object(all_objects, objects)  # latest APIs
        if opt.is_main:
            main_list = [[] for _ in range(len(objects))]
            for rank, obj_list in enumerate(all_objects):
                for i, obj in enumerate(obj_list):
                    main_list[i] += obj  # extend list to gather
            (
                tf_em,
                mc_em,
                num_char_em,
                num_float_em,
                tf_ans,
                mc_ans,
                num_char_ans,
                num_float_ans,
                global_logits,
                question_ids,
                gt_answers,
            ) = main_list
            logger.info(stats_before_gather)
            logger.info(
                "Total number of DPs after gather: {:d}. Mode: {:s}".format(
                    len(tf_em) + len(mc_em) + len(num_char_em), mode
                )
            )
    else:
        logger.info(
            "Inference for {:s} set at epoch {:d} is done (with {:d} examples).".format(
                mode, epoch, len(question_ids)
            )
        )

    if mode in ["train", "eval"] and (not opt.is_distributed or opt.is_main):
        str_mode = "TRAIN" if mode == "train" else "EVAL"
        if len(tf_em) == 0:
            logger.info(f"{str_mode:10}For T/F: Predicted N/A")
        else:
            logger.info(
                f"{str_mode:10}For T/F: Predicted {tf_em.count(1)} Match {tf_em.count(0)} Wrong ({tf_ans.count('yes')} YES {tf_ans.count('no')} NO) {'':6} | EM: {round(tf_em.count(1) / len(tf_em) * 100, 2)}"
            )
        if len(mc_em) == 0:
            logger.info(f"{'':10}For MC : Predicted N/A")
        else:
            logger.info(
                f"{'':10}For MC : Predicted {mc_em.count(1)} Match {mc_em.count(0)} Wrong {'':18} | EM: {round(mc_em.count(1) / len(mc_em) * 100, 2)}"
            )
        if len(num_char_em) == 0:
            logger.info(f"{'':10}For NUM: Predicted N/A")
        else:
            logger.info(
                f"{'':10}For NUM: Predicted (binned) {num_char_em.count(1)} Match {num_char_em.count(0)} Wrong {num_float_ans.count(None)} Invalid {'':10} | Binned EM: {round(num_char_em.count(1) / len(num_char_em) * 100, 2)}"
            )
            logger.info(
                f"{'':10}For NUM: Predicted (float) | MAE: {round(np.mean(num_float_em), 2)}"
            )

        if tb_logger is not None:
            tb_logger.add_scalar(
                f"Inference@{str_mode}/tf_em",
                np.mean(tf_em) if len(tf_em) else 0.0,
                epoch,
            )
            tb_logger.add_scalar(
                f"Inference@{str_mode}/mc_em",
                np.mean(mc_em) if len(mc_em) else 0.0,
                epoch,
            )
            tb_logger.add_scalar(
                f"Inference@{str_mode}/num_char_em",
                np.mean(num_char_em) if len(num_char_em) else 0.0,
                epoch,
            )
            tb_logger.add_scalar(
                f"Inference@{str_mode}/num_float_em",
                np.mean(num_float_em) if len(num_float_em) else 0.0,
                epoch,
            )

    if mode == "eval" and (not opt.is_distributed or opt.is_main):
        with open(checkpoint_path / f"results_epoch{epoch}.obj", "wb") as f:
            pickle.dump(list(zip(question_ids, gt_answers, global_logits)), f)

    global_em, total_questions = src.util.weighted_average(
        np.mean(global_em), total_questions, opt
    )
    return global_em


def init_basics():
    """
    Initialize basic settings for the training
    """
    # init
    opt = arg_parser()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    torch.manual_seed(
        opt.global_rank + opt.seed
    )  # different seed for different sampling depending on global_rank

    # add exp comment and timestamp to log folder name
    if len(opt.comment):
        opt.name += f"_{opt.comment}"
    opt.name = f"{opt.name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    opt.checkpoint_path = checkpoint_path
    opt.checkpoint_exists = checkpoint_exists

    # log to file
    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        (
            checkpoint_path / "run.log"
            if not opt.is_distributed
            else checkpoint_path / f"run_rank_{opt.local_rank}.log"
        ),
    )
    logger.info("**********************Start logging**********************")
    gpu_list = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
        else "ALL"
    )
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)
    for key, val in vars(opt).items():
        logger.info("{:16} {}".format(key, val))

    return opt, logger


def init_model(opt, logger):
    """
    Initialize the model
    """
    model_name = "t5-" + opt.model_size
    model_class = src.model_multihead.FiDT5

    if not opt.checkpoint_exists and opt.model_path == "none":
        # build model from scratch
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir="huggingface_cache", local_files_only=True
        )
        t5.config.dropout_rate = opt.dropout
        model = model_class(t5.config, logger)
        model.load_t5_multihead(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        # load the latest checkpoint
        load_path = opt.checkpoint_path / "checkpoint" / "latest"
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(
            model_class, load_path, opt, reset_params=False
        )
        logger.info(f"Model (latest checkpoint) loaded from {load_path}")
    else:
        # load the specified checkpoint
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(
            model_class, opt.model_path, opt, reset_params=True
        )
        logger.info(f"Model (specified checkpoint) loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)  # set pytorch checkpoint flag

    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_name,
        model_max_length=1e6,
        cache_dir="huggingface_cache",
        local_files_only=True,
    )
    collator = src.data_loader.Collator(
        opt.text_maxlength,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        n_context=opt.n_context,
    )
    return model, optimizer, scheduler, step, best_dev_em, tokenizer, collator


def init_ddp(opt, logger, model, train_dataset, eval_dataset):
    """
    Initialize DDP
    """
    if opt.is_distributed:
        logger.info(
            "Distributed ON. Mode: DDP. Backend: {:s}, Rank: {:d} / World size: {:d}. "
            "Current device: {}, spec: {}".format(
                dist.get_backend(),
                dist.get_rank(),
                dist.get_world_size(),
                torch.cuda.current_device(),
                torch.cuda.get_device_name(),
            )
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    if opt.is_distributed:
        logger.info(
            "Rank: {:d}, Training set size: {:d}, Testing set size: {:d}".format(
                dist.get_rank(), len(train_dataset), len(eval_dataset)
            )
        )
    else:
        logger.info(
            "Training set size: {:d}, Testing set size: {:d}".format(
                len(train_dataset), len(eval_dataset)
            )
        )
    return model


def main():
    """
    Main function to start the training
    """

    """options and data initialization"""
    opt, logger = init_basics()
    train_dataset, eval_dataset = src.data_loader.load_datasets(opt, logger)

    """model initialization"""
    model, optimizer, scheduler, step, best_dev_em, tokenizer, collator = init_model(
        opt, logger
    )

    """DDP initialization"""
    model = init_ddp(opt, logger, model, train_dataset, eval_dataset)

    """start training"""
    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        tokenizer,
        collator,
        best_dev_em,
        opt.checkpoint_path,
        logger,
    )

    """start evaluation"""
    logger.info("Start evaluating")
    evaluate(
        model,
        eval_dataset,
        tokenizer,
        collator,
        opt,
        opt.epochs + 1,
        logger,
        opt.checkpoint_path,
    )

    """exit DDP"""
    if opt.is_distributed:
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
