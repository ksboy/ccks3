MAX_LENGTH=256
TASK=role
DOMAIN=trans
MODEL=/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
# MODEL=./output/trigger_base/0/
DATA_DIR=./data/FewFC-main/rearranged/$DOMAIN/0/
SCHEMA=./data/event_schema/$DOMAIN.json
OUTPUT_DIR=./output/bin_multi_task/0/
BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=45
SAVE_STEPS=100
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=100
SEED=1
LR=3e-5

# CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_bi_ner_joint.py \
CUDA_VISIBLE_DEVICES=0 python3 run_bi_ner_multi_task.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_train \
--do_eval \
--do_predict \
--evaluate_during_training \
--data_dir $DATA_DIR \
--do_lower_case \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--save_steps $SAVE_STEPS \
--logging_steps $SAVE_STEPS \
--num_train_epochs $NUM_EPOCHS \
--early_stop 3 \
--learning_rate $LR \
--weight_decay 0 \
--warmup_steps $WARMUP_STEPS \
--seed $SEED \
--overwrite_output_dir \
--overwrite_cache > $OUTPUT_DIR/output.log 2>&1 &
# --fp16 \
# --freeze 
# --eval_all_checkpoints \


