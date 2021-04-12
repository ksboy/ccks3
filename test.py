# from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification
# from transformers import glue_convert_examples_to_features
# tokenizer = BertTokenizerFast.from_pretrained("/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/" ,\
#     tokenize_chinese_chars=True)
# model = BertForSequenceClassification.from_pretrained("/home/whou/workspace/pretrained_models/chinese_wwm_ext_pytorch/")
# res= tokenizer.encode_plus("我是侯 伟", return_offsets_mapping=True)
# print(res)

# from scipy.special import softmax
# aList = [2.3, -2.3]
# print(softmax(aList))

# alist = [1,2,3]
# bList= [2,3,4]
# for a in alist:
#     print(a)
#     if a in bList:
#         continue

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

lstm = nn.LSTM(1, 1)
input_x = [[1,2,3],[1,2],[1]]
seq_lens = [len(x) for x in input_x]
input_x = pad_sequence([torch.tensor(x) for x in input_x], batch_first=True).float().unsqueeze(-1)
input_x_packed = pack_padded_sequence(input_x, seq_lens, batch_first=True)
outputs_packed, _ = lstm(input_x_packed)
outputs, seq_lens = pad_packed_sequence(outputs_packed, batch_first=True)
print(outputs, seq_lens)
