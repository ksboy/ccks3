from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/" ,\
    tokenize_chinese_chars=True)
res= tokenizer.tokenize("我是侯 伟")
print(res)

# from sklearn.metrics import f1_score, precision_score, recall_score

# preds = [(1,2,3), (3,4,5)]
# labels = [(2,3,4), (3,4,5)]

# print(f1_score(labels, preds, average='macro'))