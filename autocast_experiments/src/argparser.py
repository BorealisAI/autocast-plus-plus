# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="PyTorch Autocast Experiments")

    # optim options
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument(
        "--scheduler_steps",
        type=int,
        default=None,
        help="total number of step for the scheduler, if None then scheduler_total_step = total_step",
    )
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="fixed")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--fixed_lr", action="store_true")
    parser.add_argument("--loss_reweight", action="store_true")

    # reader options
    parser.add_argument(
        "--train_data", type=str, default="none", help="path of train data"
    )
    parser.add_argument(
        "--eval_data", type=str, default="none", help="path of eval data"
    )
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument(
        "--use_checkpoint", action="store_true", help="use checkpoint in the encoder"
    )
    parser.add_argument(
        "--text_maxlength",
        type=int,
        default=200,
        help="maximum number of tokens in text segments (question+passage)",
    )
    parser.add_argument(
        "--answer_maxlength",
        type=int,
        default=-1,
        help="maximum number of tokens used to train the model, no truncation if -1",
    )
    parser.add_argument("--n_context", type=int, default=1)

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Threshold for news articles retrieval score",
    )
    parser.add_argument(
        "--train_with_news",
        action="store_true",
        help="train the model with questions that come with news articles",
    )

    # basic parameters
    parser.add_argument(
        "--name", type=str, default="experiment_name", help="name of the experiment"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoint/",
        help="models are saved here",
    )
    parser.add_argument(
        "--model_path", type=str, default="none", help="path for retraining"
    )
    parser.add_argument(
        "--comment", type=str, default="", help="comment for the experiment"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")

    # dataset parameters
    parser.add_argument(
        "--per_gpu_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--main_port",
        type=int,
        default=-1,
        help="Main port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )

    # training parameters
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5,
        help="save model every <save_freq> epochs during training",
    )

    args = parser.parse_args()
    return args


def get_deepspeed_config(
    bs_per_gpu=1,
    lr=1e-4,
    accumulation_steps=1,
):
    """
    Returns a dictionary containing the deepspeed configuration.
    """
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": bs_per_gpu,
        "gradient_accumulation_steps": accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {"lr": lr}},
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": 1000,
                "warmup_min_ratio": 0,
                "warmup_num_steps": 50,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 1e9,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": False,
        },
    }
    return deepspeed_config
