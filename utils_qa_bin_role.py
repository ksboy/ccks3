# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import json
from utils import get_labels, write_file

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, id, words, event_type, role, start_labels, end_labels):
        """Constructs a InputExample.

        Args:
            id: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.words = words
        self.event_type = event_type
        self.role = role
        self.start_labels = start_labels
        self.end_labels = end_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, start_label_ids, end_label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_label_ids = start_label_ids
        self.end_label_ids = end_label_ids

## ccks格式
def role_process_bin_ccks(input_file, schema_file, is_predict=False):
    role_dict = {}
    rows = open(schema_file, encoding='utf-8').read().splitlines()
    for row in rows:
        row = json.loads(row)
        event_type = row['event_type']
        role_dict[event_type] = []
        for role in row["role_list"]:
            role_dict[event_type].append(role["role"])

    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    count = 0
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        count += 1
        if "id" not in row:
            row["id"]=count
        # arguments = []
        if is_predict: 
            results.append({"id":row["id"], "words":list(row["content"]), "start_labels":start_labels, "end_labels":end_labels})
            continue
        for event in row["events"]:
            event_type = event["type"]
            for gold_role in role_dict[event_type]:
                start_labels = [0]*len(row["content"]) 
                end_labels = [0]*len(row["content"]) 
                for arg in event["mentions"]:
                    role = arg['role']
                    if role=="trigger": continue
                    if role!=gold_role: continue
                    argument_start_index, argument_end_index = arg["span"]
                    argument_end_index -= 1
                    start_labels[argument_start_index] = 1
                    end_labels[argument_end_index] = 1 

                results.append({"id":row["id"], "words":list(row["content"]), "event_type":event_type, "role":gold_role, \
                    "start_labels":start_labels, "end_labels":end_labels})
    return results

## lic格式
def role_process_bin_lic(schema_file, input_file, is_predict=False):
    role_dict = {}
    rows = open(schema_file, encoding='utf-8').read().splitlines()
    for row in rows:
        row = json.loads(row)
        event_type = row['event_type']
        role_dict[event_type] = []
        for role in row["role_list"]:
            role_dict[event_type].append(role["role"])

    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    count = 0
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        count += 1
        if "id" not in row:
            row["id"]=count
        # arguments = []
        if is_predict: 
            results.append({"id":row["id"], "words":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels})
            continue
        for event in row["event_list"]:
            event_type = event["type"]
            for gold_role in role_dict[event_type]:
                start_labels = ['O']*len(row["text"]) 
                end_labels = ['O']*len(row["text"])
                for arg in event["arguments"]:
                    role = arg['role']
                    if role!=gold_role: continue
                    argument = arg['argument']
                    argument_start_index = arg["argument_start_index"]
                    argument_end_index = argument_start_index + len(argument) -1
                    start_labels[argument_start_index] = 1
                    end_labels[argument_end_index] = 1 

                results.append({"id":row["id"], "words":list(row["content"]), "event_type":event_type, "role":gold_role, \
                    "start_labels":start_labels, "end_labels":end_labels})
    return results

def read_examples_from_file(data_dir, schema_file, mode, task, dataset="ccks"):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    if dataset=="ccks":
        # if task=='trigger': items = trigger_process_bin_ccks(file_path)
        if task=='role': items = role_process_bin_ccks(file_path, schema_file,)
    elif dataset=="lic":
        # if task=='trigger': items = trigger_process_bin_lic(file_path)
        if task=='role': items = role_process_bin_lic(file_path, schema_file,)
    return [InputExample(**item) for item in items]

def get_query_templates(query_file):
    """Load query templates"""
    query_templates = dict()
    with open(query_file, "r", encoding='utf-8') as f:
        for line in f:
            event_type, role, description, query = line.strip().split(",")
            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if role not in query_templates[event_type]:
                query_templates[event_type][role] = list()

            # 0 template role
            query_templates[event_type][role].append(role)
            # 1 template role + in trigger (replace [trigger] when forming the instance)
            query_templates[event_type][role].append(role + " in [trigger]")
            # 2 template arg_query
            query_templates[event_type][role].append(query)
            # 3 arg_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][role].append(query[:-1] + " in [trigger]?")
    return query_templates

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
    nth_query=2,
    dataset='ccks',
    task='trigger'
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        # print(example.words, example.labels)
        # print(len(example.words), len(example.labels))
        tokens = []
        start_label_ids = []
        end_label_ids = []
        token_type_ids = []
        
        # query
        query_templates = get_query_templates("./query_template/"+dataset+".csv")
        event_type, role = example.event_type, example.role
        query = query_templates[event_type][role][nth_query]

        for i in range(len(query)):
            word = query[i]
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens)==1:
                tokens.extend(word_tokens)
            if len(word_tokens)>1: 
                print(word,">1") 
                tokens.extend(word_tokens[:1])
                pass
            if len(word_tokens)<1: 
                # print(word,"<1") 基本都是空格
                tokens.extend(["[unused1]"])
                # continue
            start_label_ids.append(pad_token_label_id)
            end_label_ids.append(pad_token_label_id)
        
        # [SEP]
        tokens += [sep_token]
        start_label_ids += [pad_token_label_id]
        end_label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # paragraph
        for word, start_label, end_label in zip(example.words, example.start_labels, example.end_labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens)==1:
                tokens.extend(word_tokens)
            if len(word_tokens)>1: 
                print(word,">1") 
                tokens.extend(word_tokens[:1])
                pass
            if len(word_tokens)<1: 
                # print(word,"<1") 基本都是空格
                tokens.extend(["[unused1]"])
                # continue

            start_label_ids.append(start_label)
            end_label_ids.append(end_label)

            token_type_ids.append(sequence_b_segment_id)
            # if len(tokens)!= len(label_ids):
            #     print(word, word_tokens, tokens, label_ids)
        # print(len(tokens),len(label_ids))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_label_ids = start_label_ids[: (max_seq_length - special_tokens_count)]
            end_label_ids = end_label_ids[: (max_seq_length - special_tokens_count)]
            token_type_ids = token_type_ids[: (max_seq_length - special_tokens_count)]
        
        # [SEP]
        tokens += [sep_token]
        start_label_ids += [pad_token_label_id]
        end_label_ids += [pad_token_label_id]
        token_type_ids += [sequence_b_segment_id]


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            start_label_ids += [pad_token_label_id]
            end_label_ids += [pad_token_label_id]
            token_type_ids += [sequence_b_segment_id]

        if cls_token_at_end:
            tokens += [cls_token]
            start_label_ids += [pad_token_label_id]
            end_label_ids += [pad_token_label_id]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_label_ids = [pad_token_label_id] + start_label_ids
            end_label_ids = [pad_token_label_id] + end_label_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(len(tokens), len(input_ids), len(label_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            start_label_ids = ([pad_token_label_id] * padding_length) + start_label_ids
            end_label_ids = ([pad_token_label_id] * padding_length) + end_label_ids
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            start_label_ids += [pad_token_label_id] * padding_length
            end_label_ids += [pad_token_label_id] * padding_length
        
        # print(len(label_ids), max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(start_label_ids) == max_seq_length
        assert len(end_label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s", example.id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))
            logger.info("start_label_ids: %s", " ".join([str(x) for x in start_label_ids]))
            logger.info("end_label_ids: %s", " ".join([str(x) for x in end_label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                start_label_ids=start_label_ids, end_label_ids= end_label_ids)
        )
    return features

