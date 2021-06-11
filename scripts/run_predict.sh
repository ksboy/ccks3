MAX_LENGTH=256
DATASET=ccks
TASK=trigger
DOMAIN=base
MODEL=/hy-nas/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/  #albert-xxlarge-v2/  #bert-large-uncased-wwm/
DATA_DIR=./data/FewFC-main/rearranged/$DOMAIN/
SCHEMA=./data/FewFC-main/event_schema/$DOMAIN.json
OUTPUT_DIR=./output/ccks/base/trigger/
EVAL_BATCH_SIZE=64
SEED=1

CUDA_VISIBLE_DEVICES=0 python3 run_ner_bio.py \
--dataset $DATASET \
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