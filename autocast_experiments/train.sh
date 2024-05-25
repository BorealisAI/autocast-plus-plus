#!/bin/sh

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


export CUR_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CUR_FILE_DIR

### reader model
export READER=t5
export MODELSIZE=$1
# MODELSIZE could be 'base', 'small', '3b', etc.

### retriever option
export TOPN=$2
export SCORE_THRESHOLD=$3

### training hyperparameters
export NGPU=$4
export EPOCHS=$5
export BSZ=$6
export SCHEDULERTYPE=$7
# SCHEDULERTYPE could be 'linear', 'cosine'
export LR=$8
# LR could be 5e-5, 1e-4, etc.

export OPTIMTYPE=adamw
export WDECAY=$9
export DROPOUT=${10}

export WARMUP=0  # auto select the first epoch to warmup

export RETR=bm25ce
export TRAIN=data/static_train_top50_reorg.json
export EVAL=data/static_test_top50_reorg.json

export OMP_NUM_THREADS=6

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT


torchrun --nproc_per_node=$NGPU --master_port=$PORT train.py \
        --model_size $MODELSIZE \
        --per_gpu_batch_size $BSZ \
        --epochs $EPOCHS \
        --answer_maxlength 10 \
        --text_maxlength 512 \
        --train_data $TRAIN \
        --eval_data $EVAL \
        --n_context $TOPN \
        --name ${READER}_${MODELSIZE}_minScore${SCORE_THRESHOLD}_top${TOPN}_LRSch${SCHEDULERTYPE}_lr${LR}_WD${WDECAY}_DROP_${DROPOUT}_bs${BSZ}_ep${EPOCHS}_txtREL_ALIGN_reWeight \
        --optim $OPTIMTYPE \
        --lr $LR \
        --weight_decay $WDECAY \
        --dropout $DROPOUT \
        --scheduler $SCHEDULERTYPE \
        --warmup_steps $WARMUP \
        --train_with_news \
        --use_checkpoint \
        --seed 1 \
        --save_freq $EPOCHS \
        --score_threshold $SCORE_THRESHOLD \
        --loss_reweight \
        --comment $(hostname) 
        
