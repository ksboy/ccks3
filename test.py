# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/" ,\
#     tokenize_chinese_chars=True)
# res= tokenizer.tokenize("我是侯 伟")
# print(res)

from seqeval.metrics.sequence_labeling import get_entities
