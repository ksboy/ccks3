from transformers import BertTokenizer
from transformers import glue_convert_examples_to_features
tokenizer = BertTokenizer.from_pretrained("/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/" ,\
    tokenize_chinese_chars=True)
res= tokenizer.encode_plus("我是侯 伟")
print(res)

# from scipy.special import softmax
# aList = [2.3, -2.3]
# print(softmax(aList))
