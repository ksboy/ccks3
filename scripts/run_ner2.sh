MAX_LENGTH=256
TASK=trigger
MODEL=~/workspace/pretrained_models/chinese_roberta_wwm_large_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/trigger_trans/
SCHEMA=./data/event_schema.json
OUTPUT_DIR=./output/trigger_trans_test/
BATCH_SIZE=8
EVAL_BATCH_SIZE=64
NUM_EPOCHS=3000
SAVE_STEPS=100
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=600
SEED=1
LR=2e-5

CUDA_VISIBLE_DEVICES=0,1 python3 run_ner.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_train \
--do_eval \
--evaluate_during_training \
--data_dir $DATA_DIR \
--do_lower_case \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \
--max_seq_length  $MAX_LENGTH \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--save_steps $SAVE_STEPS \
--logging_steps $SAVE_STEPS \
--num_train_epochs $NUM_EPOCHS \
--early_stop 4 \
--learning_rate $LR \
--weight_decay 0 \
--warmup_steps $WARMUP_STEPS \
--seed $SEED 
# --overwrite_cache 
# --fp16 \
# --freeze 
# --eval_all_checkpoints \


