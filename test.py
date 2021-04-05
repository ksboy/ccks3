# from transformers import BertTokenizer, BertTokenizerFast
# from transformers import glue_convert_examples_to_features
# tokenizer = BertTokenizerFast.from_pretrained("/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/" ,\
#     tokenize_chinese_chars=True)
# res= tokenizer.encode_plus("我是侯 伟", return_offsets_mapping=True)
# print(res)

# from scipy.special import softmax
# aList = [2.3, -2.3]
# print(softmax(aList))

alist = [1,2,3]
bList= [2,3,4]
for a in alist:
    print(a)
    if a in bList:
        continue
