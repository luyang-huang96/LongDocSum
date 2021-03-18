#!/bin/sh

TOTAL_NUM_UPDATES=20000 
WARMUP_UPDATES=1000      
LR=7e-05
MAX_TOKENS=8192
UPDATE_FREQ=32
DATA_DIR=[/path/to/data/]
USER_DIR=./longbart
ARCH=longbart
SAVE_DIR=[/path/to/save/model]
TENSORBOARD_LOGDIR=[/path/to/write/log/file/]
BART_PATH=[/path/to/bart_large_ckpt]


CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA_DIR \
	    --user-dir $USER_DIR \
	    --restore-file $BART_PATH \
	    --max-tokens $MAX_TOKENS \
	    --task translation_longbart \
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
	    --weight-decay 0.01 --optimizer adafactor \
	    --clip-norm 0.1 \
	    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	    --fp16 --update-freq $UPDATE_FREQ \
	    --skip-invalid-size-inputs-valid-test \
	    --find-unused-parameters \
	    --tensorboard-logdir $TENSORBOARD_LOGDIR \
	    --save-dir $SAVE_DIR \
	    --truncate-source \
	    --max-source-positions 8192 \
	    --max-target-positions 1024 \
	    --memory-efficient-fp16 \
	    --keep-interval-updates 5 \
	    --log-interval 5 \
	    --sinkhorn \
	    --left-pad-source False \
	    --decoder_divide_attention;
