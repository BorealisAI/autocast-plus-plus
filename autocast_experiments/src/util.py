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


import os
import errno
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
import sys
import logging
import json
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # always log for different ranks
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return logger


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    if "best" not in name:
        symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path)
    model = model.to(opt.device)
    if not reset_params:
        logger.info("loading checkpoint %s" % optimizer_path)
        checkpoint = torch.load(optimizer_path, map_location=opt.device)
        opt_checkpoint = checkpoint["opt"]
        step = checkpoint["step"]
        if "best_eval_metric" in checkpoint:
            best_eval_metric = checkpoint["best_eval_metric"]
        else:
            best_eval_metric = checkpoint["best_dev_em"]
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)
        opt_checkpoint = None
        step, best_eval_metric = 0, 0.0

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        scheduler_steps,
        min_ratio,
        fixed_lr,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(
                max(1, self.warmup_steps)
            ) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(
            0.0,
            1.0
            + (self.min_ratio - 1)
            * (step - self.warmup_steps)
            / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        return 1.0


##### Credit to `github.com/katsura-jp/pytorch-cosine-annealing-with-warmup` by Naoki Katsura #####
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model, all_params=None):
    if all_params is None:
        all_params = model.parameters()
    if opt.optim == "adam":
        optimizer = torch.optim.Adam(all_params, lr=opt.lr)
    elif opt.optim == "adamw":
        optimizer = torch.optim.AdamW(
            all_params, lr=opt.lr, weight_decay=opt.weight_decay
        )

    steps_per_epoch = (
        opt.train_data_size
        // opt.per_gpu_batch_size
        // (opt.world_size if opt.is_distributed else 1)
    ) // opt.accumulation_steps
    if opt.warmup_steps == 0:
        opt.warmup_steps = steps_per_epoch  # use the first epoch to warmup

    if opt.scheduler == "fixed":
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == "linear":
        assert (
            opt.train_data_size != 0
        ), "Need to specify the training dataset size if lr decay is wanted."
        scheduler_steps = steps_per_epoch * opt.epochs
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=opt.warmup_steps,
            scheduler_steps=scheduler_steps,
            min_ratio=0.0,
            fixed_lr=opt.fixed_lr,
        )
    elif opt.scheduler == "cosine":
        assert (
            opt.train_data_size != 0
        ), "Need to specify the training dataset size if lr decay is wanted."
        scheduler_steps = steps_per_epoch * opt.epochs
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-9, end_factor=1.0, total_iters=opt.warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_steps - opt.warmup_steps, eta_min=1e-9
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [linear_scheduler, cosine_scheduler]
        )

    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        x = x.cuda()
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
        dist.barrier()
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob("*.txt"))
    files.sort()
    with open(output_path, "w") as outfile:
        for path in files:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / "tmp_dir"
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f"{opt.global_rank}.json"
    with open(tmp_path, "w") as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / "dataset_wscores.json"
        logger.info(f"Writing dataset with scores at {final_path}")
        glob_path = write_path / "*"
        results_path = write_path.glob("*.json")
        alldata = []
        for path in results_path:
            with open(path, "r") as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, "w") as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()


def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter="\t")
        for k, row in enumerate(reader):
            if not row[0] == "id":
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(
                        f"The following input line has not been correctly loaded: {row}"
                    )
    return passages


def plot_article_length(train_examples, save_path):
    # count the length of articles
    q_article_length = []
    for example in train_examples:
        if len(example["ctxs"]):
            this_article_length = []
            for ctx in example["ctxs"]:
                this_article_length.append(
                    len(ctx["title"].split()) + len(ctx["text"].split())
                )
            q_article_length.append(this_article_length)
    all_article_length = sum(q_article_length, [])
    print(
        "Statistics about length of articles. Mean: {:.2f}, Median: {:.2f}, Max: {:.2f}, Min: {:.2f}".format(
            np.mean(all_article_length),
            np.median(all_article_length),
            np.max(all_article_length),
            np.min(all_article_length),
        )
    )
    plt.hist(all_article_length, bins=1000)
    plt.xlabel("Length of articles")
    plt.ylabel("Count")
    plt.title(
        "Distribution of article length in data set.\nStatistics: Mean: {:.2f}, Median: {:.2f}, Max: {:.2f}, Min: {:.2f}".format(
            np.mean(all_article_length),
            np.median(all_article_length),
            np.max(all_article_length),
            np.min(all_article_length),
        )
    )
    plt.xlim(0, 1000)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
