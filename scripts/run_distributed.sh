MAX_LENGTH=256
DATASET=ccks
TASK=role
DOMAIN=trans
MODEL=/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
# MODEL=./output/trigger_base/0/  finetune
# DATA_DIR=./data/DuEE_1_0/
# SCHEMA=./data/DuEE_1_0/event_schema.json
# OUTPUT_DIR=./output/$DATASET/role_bin2/
DATA_DIR=./data/FewFC-main/rearranged/$DOMAIN/
SCHEMA=./data/FewFC-main/event_schema/$DOMAIN.json
OUTPUT_DIR=./output/$DATASET/$DOMAIN/multi_task_temp/
BATCH_SIZE=16
EVAL_BATCH_SIZE=64
NUM_EPOCHS=45
SAVE_STEPS=100 # 300
# SAVE_STEPS= $save_steps* gradient_accumulation_steps * batch_size * num_gpus
WARMUP_STEPS=100
SEED=1
LR=3e-5

mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 run_ner_bin_multi_task.py \
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 \
--master_addr="192.168.140.158" --master_port=1234 \
run_ner_bin_multi_task.py \
--dataset $DATASET \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--local_rank 0 \
--do_train \
--do_eval \
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
--early_stop 4 \
--learning_rate $LR \
--weight_decay 0 \
--warmup_steps $WARMUP_STEPS \
--seed $SEED \
--overwrite_output_dir \
--overwrite_cache > $OUTPUT_DIR/output.log 2>&1 &
# --fp16 \
# --freeze 
# --eval_all_checkpoints \
