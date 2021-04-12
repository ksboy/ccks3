MAX_LENGTH=256
TASK=trigger
DOMAIN=trans
MODEL=/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/FewFC-main/rearranged/$DOMAIN/0/
SCHEMA=./data/FewFC-main/event_schema/$DOMAIN.json
OUTPUT_DIR=./output/trigger_trans_bin2/0/
EVAL_BATCH_SIZE=64
SEED=1

CUDA_VISIBLE_DEVICES=0 python3 run_bi_ner.py \
--task $TASK \
--model_type bert \
--model_name_or_path $MODEL \
--do_eval \
--data_dir $DATA_DIR \
--do_lower_case \
--keep_accents \
--schema $SCHEMA \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--overwrite_cache \
--seed $SEED > $OUTPUT_DIR/eval.log 2>&1 &