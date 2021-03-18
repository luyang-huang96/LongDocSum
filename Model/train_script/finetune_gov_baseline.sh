#!/bin/sh

TOTAL_NUM_UPDATES=20000 
WARMUP_UPDATES=1000      
LR=2e-05
MAX_TOKENS=2048
UPDATE_FREQ=32
DATA_DIR=/data2/luyang/gov-report/fairseq-train/
USER_DIR=./longbart
ARCH=bart_large
SAVE_DIR=/data2/luyang/longsum/gov/baseline-wizwhy
TENSORBOARD_LOGDIR=/data2/luyang/longsum/gov/baseline-wizwhy/tensorboard/
BART_PATH=/data2/luyang/BART/bart.large/model.pt


CUDA_VISIBLE_DEVICES=2 fairseq-train $DATA_DIR \
	    --restore-file $BART_PATH \
	    --max-tokens $MAX_TOKENS \
	    --task translation \
	    --source-lang source --target-lang target \
	    --layernorm-embedding \
	    --share-all-embeddings \
	    --share-decoder-input-output-embed \
	    --reset-optimizer --reset-dataloader --reset-meters \
	    --required-batch-size-multiple 1 \
	    --arch $ARCH \
	    --criterion label_smoothed_cross_entropy \
	    --label-smoothing 0.1 \
	    --dropout 0.1 --attention-dropout 0.1 \
	    --weight-decay 0.01 --optimizer adam \
	    --clip-norm 0.1 \
	    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	    --fp16 --update-freq $UPDATE_FREQ \
	    --skip-invalid-size-inputs-valid-test \
	    --find-unused-parameters \
	    --tensorboard-logdir $TENSORBOARD_LOGDIR \
	    --save-dir $SAVE_DIR \
	    --max-source-positions 1024 \
	    --max-target-positions 1024 \
	    --memory-efficient-fp16 \
	    --keep-interval-updates 5 \
	    --log-interval 20 \
	    --truncate-source;
