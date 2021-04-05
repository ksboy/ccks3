
import logging
import os
import json
from utils import get_labels
logger = logging.getLogger(__name__)

candidate_queries = [
['what', 'is', 'the', 'trigger', 'in', 'the', 'event', '?'], # 0 what is the trigger in the event?
['what', 'happened', 'in', 'the', 'event', '?'], # 1 what happened in the event?
['trigger'], # 2 trigger
['t'], # 3 t
['action'], # 4 action
['verb'], # 5 verb
['null'], # 6 null
]
                    

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 # unique_id,
                 # example_index,
                 # doc_span_index,
                 sentence_id,
                 tokens,
                 # token_to_orig_map,
                 # token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 in_sentence,
                 labels):
        # self.unique_id = unique_id
        # self.example_index = example_index
        # self.doc_span_index = doc_span_index
        self.sentence_id = sentence_id
        self.tokens = tokens
        # self.token_to_orig_map = token_to_orig_map
        # self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.in_sentence = in_sentence
        self.labels = labels


def read_ace_examples(nth_query, input_file, tokenizer, category_vocab, is_training):
    """Read an ACE json file, transform to features"""
    features = []
    examples = []
    sentence_id = 0
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            sentence, events, s_start = example["sentence"], example["event"], example["s_start"]
            offset_category = dict()
            for event in events:
                assert len(event[0]) == 2
                offset, category = event[0][0] - s_start, event[0][1]
                offset_category[offset] = category

            tokens = []
            segment_ids = []
            in_sentence = []
            labels = []

            # add [CLS]
            tokens.append("[CLS]")
            segment_ids.append(0)
            in_sentence.append(0)
            labels.append(category_vocab.category_to_index["None"])

            # add query
            query = candidate_queries[nth_query]
            for (i, token) in enumerate(query):
                sub_tokens = tokenizer.tokenize(token)
                tokens.append(sub_tokens[0])
                segment_ids.append(0)
                in_sentence.append(0)
                labels.append(category_vocab.category_to_index["None"])

            # add [SEP]
            tokens.append("[SEP]")
            segment_ids.append(0)
            in_sentence.append(0)
            labels.append(category_vocab.category_to_index["None"])

            # add sentence
            for (i, token) in enumerate(sentence):
                sub_tokens = tokenizer.tokenize(token)
                tokens.append(sub_tokens[0])
                segment_ids.append(1)
                in_sentence.append(1)
                if i in offset_category:
                    labels.append(category_vocab.category_to_index[offset_category[i]])
                else:
                    labels.append(category_vocab.category_to_index["None"])

            # add [SEP]
            tokens.append("[SEP]")
            segment_ids.append(1)
            in_sentence.append(0)
            labels.append(category_vocab.category_to_index["None"])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < category_vocab.max_sent_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                in_sentence.append(0)
                labels.append(category_vocab.category_to_index["None"])

            # print(len(input_ids), category_vocab.max_sent_length)
            assert len(input_ids) == category_vocab.max_sent_length
            assert len(segment_ids) == category_vocab.max_sent_length
            assert len(in_sentence) == category_vocab.max_sent_length
            assert len(input_mask) == category_vocab.max_sent_length
            assert len(labels) == category_vocab.max_sent_length

            features.append(
                InputFeatures(
                    # unique_id=unique_id,
                    # example_index=example_index,
                    sentence_id=sentence_id,
                    tokens=tokens,
                    # token_to_orig_map=token_to_orig_map,
                    # token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    in_sentence=in_sentence,
                    labels=labels))
            examples.append(example)
            # if len(tokens) > 20 and sum(labels) > 0:
                # import ipdb; ipdb.set_trace()
            sentence_id += 1

    return examples, features   


import numpy as np
def get_entities(start_logits, end_logits, attention_mask=None):
    # start_logits: [batch_size, seq_length, labels]
    if attention_mask is None:
        attention_mask = np.ones(start_logits.shape[:-1])
    batch_size, seq_length, num_labels = start_logits.shape
    batch_pred_list = []
    dis = 12
    for i in range(batch_size):   # batch_index
        cur_pred_list=[]
        for j in range(seq_length):  # token_index 
            if not attention_mask[i, j]: continue
            # 实体 头
            for k in range(num_labels):  
                if start_logits[i][j][k]:
                    # 寻找 实体尾 
                    for l in range(j, min(j+ dis, seq_length)):
                        if end_logits[i][l][k]:
                            cur_pred_list.append((i, j, l, k)) # index, start, end, label
                            break
        batch_pred_list.append(cur_pred_list)
    return batch_pred_list

