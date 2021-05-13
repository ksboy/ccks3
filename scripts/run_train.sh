MAX_LENGTH=256
DATASET=ccks
TASK=role
DOMAIN=base
MODEL=/home/whou/workspace/pretrained_models/chinese_ro_bert_a_wwm_large_ext_pytorch/  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
# MODEL=./output/trigger_base/0/  finetune
# DATA_DIR=./data/DuEE_1_0/
# SCHEMA=./data/DuEE_1_0/event_schema.json
# OUTPUT_DIR=./output/$DATASET/role_bin2_with_gate_whou_roberta_large_relu/
DATA_DIR=./data/FewFC-main/rearranged/$DOMAIN/
SCHEMA=./data/FewFC-main/event_schema/$DOMAIN.json
OUTPUT_DIR=./output/$DATASET/$DOMAIN/role_qa_roberta_large/query_3/
BATCH_SIZE=3
# BATCH_SIZE = batch_size * num_gpus
EVAL_BATCH_SIZE=64
NUM_EPOCHS=45
SAVE_STEPS=300 # 100
# SAVE_STEPS= save_steps * gradient_accumulation_steps
WARMUP_STEPS=100
SEED=1
LR=3e-5

mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0,1 nohup python -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_qa_bin_role.py \
# CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m torch.distributed.launch --nproc_per_node=2 run_qa_bin_role.py --local_rank 0 \
CUDA_VISIBLE_DEVICES=0 nohup python -u run_qa_bin_role.py \
--dataset $DATASET \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_train \
--do_eval \
--evaluate_during_training \
--data_dir $DATA_DIR \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps 4 \
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

