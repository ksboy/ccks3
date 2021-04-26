from transformers import BertTokenizer, BertTokenizerFast, BertModel, BertForSequenceClassification, BertForQuestionAnswering
# from transformers import glue_convert_examples_to_features
tokenizer = BertTokenizer.from_pretrained("/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/" ,\
    tokenize_chinese_chars=True)
input = tokenizer.batch_encode_plus([["我是","你好"]], max_length=10, pad_to_max_length=True, return_tensors='pt')
print(input)
model = BertModel.from_pretrained("/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/")
output = model(**input)
print(output[0])

# from scipy.special import softmax
# aList = [2.3, -2.3]
# print(softmax(aList))

# alist = [1,2,3]
# bList= [2,3,4]
# for a in alist:
#     print(a)
#     if a in bList:
#         continue

# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# lstm = nn.LSTM(1, 1)
# input_x = [[1,2,3],[1,2],[1]]
# seq_lens = [len(x) for x in input_x]
# input_x = pad_sequence([torch.tensor(x) for x in input_x], batch_first=True).float().unsqueeze(-1)
# input_x_packed = pack_padded_sequence(input_x, seq_lens, batch_first=True)
# outputs_packed, _ = lstm(input_x_packed)
# outputs, seq_lens = pad_packed_sequence(outputs_packed, batch_first=True)
# print(outputs, seq_lens)

# import json
# a= {1:'1'}
# file = open('1.json','w',encoding='utf-8')
# json.dump(a,file)


# import json
# infile = open("./data/ACE05/dev_process.json").read()
# tmp = json.loads(infile)
# print(tmp)