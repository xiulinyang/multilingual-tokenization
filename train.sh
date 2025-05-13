#!/bin/bash

LANG=$1
VOCAB_SIZE=$2
SEED=$3


MODEL_NAME=MODEL_NAME=${LANG}_${VOCAB_SIZE}_${SEED}

python tokenizer_and_config.py -m "$MODEL_NAME" \
    --bpe \
    --vocab "$VOCAB_SIZE" \
    --train_file "/scratch/xiulyang/multilingual-LM/data/multilingual/$LANG/train/$LANG.train" \

# warning: this will upload to your huggingface account (as long as you have the token set up)
# else, it will error out; just omit --push_to_hub if you do not want this.

python train_autoreg.py \
    --config_name models/$MODEL_NAME \
    --tokenizer_name models/$MODEL_NAME \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --train_file /scratch/xiulyang/multilingual-LM/data/multilingual/$LANG/train/$LANG.train \
    --validation_file /scratch/xiulyang/multilingual-LM/data/multilingual/$LANG/dev/$LANG.dev \
    --evaluation_strategy epoch \
    --output_dir models/$MODEL_NAME \
    --overwrite_output_dir \
    --learning_rate 0.0006 \
    --save_strategy epoch\
    --load_best_model_at_end True \
    --block_size 512 \
    --num_train_epochs 10 \
    --logging_steps 1000 \
    --add_prefix_space \
    --warmup_steps 1000 \
    --seed $SEED \
    --fp16 \
    --weight_decay 0.1\
    --report_to wandb \
#    --push_to_hub \
#    --hub_model_id $MODEL_NAME