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

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, id,  words, token_type_ids, \
                trigger_start_labels,trigger_end_labels,role_start_labels,role_end_labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.words = words
        self.token_type_ids = token_type_ids # 目标trigger位置
        self.trigger_start_labels = trigger_start_labels
        self.trigger_end_labels = trigger_end_labels
        self.role_start_labels = role_start_labels
        self.role_end_labels = role_end_labels

## ccks格式
def data_process_bin_ccks(input_file, add_event_type_to_role=True, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        
        if is_predict:
            results.append({"id":row["id"], "tokens":list(row["content"]), "token_type_ids": [0] * len(row["content"]), \
              "trigger_start_labels":['O']*len(row["content"]), "trigger_end_labels":['O']*len(row["content"]), \
              "role_start_labels":['O']*len(row["content"]), "role_end_labels":['O']*len(row["content"])})
            continue
        
        # trigger
        trigger_start_labels = ['O']*len(row["content"]) 
        trigger_end_labels = ['O']*len(row["content"]) 
        for event in row["events"]:
            event_type = event["type"]
            for arg in event["mentions"]:
                role = arg['role']
                # trigger
                if role=="trigger":
                    trigger_start_index, trigger_end_index = arg["span"]
                    trigger_end_index -= 1
                    if trigger_start_labels[trigger_start_index]=="O":
                        trigger_start_labels[trigger_start_index] = event_type
                    else: 
                        trigger_start_labels[trigger_start_index] += (" "+ event_type)
                    if trigger_end_labels[trigger_end_index]=="O":
                        trigger_end_labels[trigger_end_index] = event_type
                    else: 
                        trigger_end_labels[trigger_end_index] += (" "+ event_type)
                    break
        
        # role
        role_start_labels = ['O']*len(row["content"]) 
        role_end_labels = ['O']*len(row["content"])
        for event in row["events"]:
            event_type = event["type"]
            token_type_ids= [0] * len(row["content"])
            for arg in event["mentions"]:
                role = arg['role']
                # token_type_ids
                if role=="trigger":
                    for i in range(trigger_start_index, trigger_end_index+1):
                        token_type_ids[i] = 1
                    continue
                if add_event_type_to_role: role = event_type + '-' + role
                argument_start_index, argument_end_index = arg["span"]
                argument_end_index -= 1
                if role_start_labels[argument_start_index]=="O":
                    role_start_labels[argument_start_index] = role
                else: 
                    role_start_labels[argument_start_index] += (" "+ role)
                    
                if role_end_labels[argument_end_index]=="O":
                    role_end_labels[argument_end_index] = role
                else: 
                    role_end_labels[argument_end_index] += (" "+ role)

            results.append({"id":row["id"], "words":list(row["content"]), "token_type_ids":token_type_ids, \
                "trigger_start_labels":trigger_start_labels, "trigger_end_labels":trigger_end_labels, \
                    "role_start_labels":role_start_labels, "role_end_labels":role_end_labels})
    return results

# lic格式
def data_process_bin_lic(input_file, add_event_type_to_role=True, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        
        if is_predict:
            results.append({"id":row["id"], "tokens":list(row["text"]), "token_type_ids": [0] * len(row["text"]), \
              "trigger_start_labels":['O']*len(row["text"]), "trigger_end_labels":['O']*len(row["text"]), \
              "role_start_labels":['O']*len(row["text"]), "role_end_labels":['O']*len(row["text"])})
            continue
        
        # trigger
        trigger_start_labels = ['O']*len(row["text"]) 
        trigger_end_labels = ['O']*len(row["text"]) 
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event['trigger_start_index']
            trigger_end_index = trigger_start_index + len(trigger) -1
            if trigger_start_labels[trigger_start_index]=="O":
                trigger_start_labels[trigger_start_index] = event_type
            else: 
                trigger_start_labels[trigger_start_index] += (" "+ event_type)
            if trigger_end_labels[trigger_end_index]=="O":
                trigger_end_labels[trigger_end_index] = event_type
            else: 
                trigger_end_labels[trigger_end_index] += (" "+ event_type)
        
        # role
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event['trigger_start_index']
            token_type_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                token_type_ids[i] = 1

            role_start_labels = ['O']*len(row["text"]) 
            role_end_labels = ['O']*len(row["text"]) 

            for arg in event["arguments"]:
                role = arg['role']
                if add_event_type_to_role: role = event_type + '-' + role
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1
                
                if role_start_labels[argument_start_index]=="O":
                    role_start_labels[argument_start_index] = role
                else: 
                    role_start_labels[argument_start_index] += (" "+ role)
                    
                if role_end_labels[argument_end_index]=="O":
                    role_end_labels[argument_end_index] = role
                else: 
                    role_end_labels[argument_end_index] += (" "+ role)

                # if arg['alias']!=[]: print(arg['alias'])
            results.append({"id":row["id"], "words":list(row["text"]), "token_type_ids":token_type_ids, \
                "trigger_start_labels":trigger_start_labels, "trigger_end_labels":trigger_end_labels, \
                    "role_start_labels":role_start_labels, "role_end_labels":role_end_labels})
    return results

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, \
            trigger_start_label_ids, trigger_end_label_ids,\
            role_start_label_ids, role_end_label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_start_label_ids = trigger_start_label_ids
        self.trigger_end_label_ids = trigger_end_label_ids
        self.role_start_label_ids = role_start_label_ids
        self.role_end_label_ids = role_end_label_ids


def read_examples_from_file(data_dir, mode, dataset="ccks"):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    if dataset=="ccks":
        items = data_process_bin_ccks(file_path, add_event_type_to_role=False, is_predict= mode!='train')
    elif dataset=="lic":
        items = data_process_bin_lic(file_path, add_event_type_to_role=False, is_predict= mode!='train')
    return [InputExample(**item) for item in items]

def convert_examples_to_features(
    examples,
    trigger_label_list,
    role_label_list,
    max_seq_length,
    tokenizer,
    trigger_token_segment_id = 1,
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
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    trigger_label_map = {label: i for i, label in enumerate(trigger_label_list)}
    trigger_label_map['O'] = -1
    role_label_map = {label: i for i, label in enumerate(role_label_list)}
    role_label_map['O'] = -1
    # print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        # print(example.words, example.labels)
        # print(len(example.words), len(example.labels))
        tokens = []
        trigger_start_label_ids = []
        trigger_end_label_ids = []
        role_start_label_ids = []
        role_end_label_ids = []
        token_type_ids = []
        for word, segment_id, trigger_start_label, trigger_end_label, role_start_label, role_end_label \
            in zip(example.words,  example.token_type_ids, \
                example.trigger_start_labels, example.trigger_end_labels, \
                example.role_start_labels, example.role_end_labels):
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
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            ##################################################
            # trigger 
            cur_start_labels = trigger_start_label.split()
            cur_start_label_ids = []
            for cur_start_label in cur_start_labels:
                cur_start_label_ids.append(trigger_label_map[cur_start_label])
            trigger_start_label_ids.append(cur_start_label_ids)

            cur_end_labels = trigger_end_label.split()
            cur_end_label_ids = []
            for cur_end_label in cur_end_labels:
                cur_end_label_ids.append(trigger_label_map[cur_end_label])
            trigger_end_label_ids.append(cur_end_label_ids)

            ##################################################
            # role 
            cur_start_labels = role_start_label.split()
            cur_start_label_ids = []
            for cur_start_label in cur_start_labels:
                cur_start_label_ids.append(role_label_map[cur_start_label])
            role_start_label_ids.append(cur_start_label_ids)

            cur_end_labels = role_end_label.split()
            cur_end_label_ids = []
            for cur_end_label in cur_end_labels:
                cur_end_label_ids.append(role_label_map[cur_end_label])
            role_end_label_ids.append(cur_end_label_ids)


            token_type_ids.extend( [sequence_a_segment_id if not segment_id else trigger_token_segment_id] )


            # if len(tokens)!= len(label_ids):
            #     print(word, word_tokens, tokens, label_ids)
        # print(len(tokens),len(label_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            trigger_start_label_ids = trigger_start_label_ids[: (max_seq_length - special_tokens_count)]
            trigger_end_label_ids = trigger_end_label_ids[: (max_seq_length - special_tokens_count)]
            role_start_label_ids = role_start_label_ids[: (max_seq_length - special_tokens_count)]
            role_end_label_ids = role_end_label_ids[: (max_seq_length - special_tokens_count)]
            token_type_ids = token_type_ids[: (max_seq_length - special_tokens_count)]


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
        tokens += [sep_token]
        trigger_start_label_ids += [[pad_token_label_id]]
        trigger_end_label_ids += [[pad_token_label_id]]
        role_start_label_ids += [[pad_token_label_id]]
        role_end_label_ids += [[pad_token_label_id]]
        token_type_ids += [sequence_a_segment_id]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            trigger_start_label_ids += [[pad_token_label_id]]
            trigger_end_label_ids += [[pad_token_label_id]]
            role_start_label_ids += [[pad_token_label_id]]
            role_end_label_ids += [[pad_token_label_id]]
            token_type_ids += [sequence_a_segment_id]

        if cls_token_at_end:
            tokens += [cls_token]
            trigger_start_label_ids += [[pad_token_label_id]]
            trigger_end_label_ids += [[pad_token_label_id]]
            role_start_label_ids += [[pad_token_label_id]]
            role_end_label_ids += [[pad_token_label_id]]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            trigger_start_label_ids = [[pad_token_label_id]] + trigger_start_label_ids
            trigger_end_label_ids = [[pad_token_label_id]] + trigger_end_label_ids
            role_start_label_ids = [[pad_token_label_id]] + role_start_label_ids
            role_end_label_ids = [[pad_token_label_id]] + role_end_label_ids
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
            trigger_start_label_ids = ([[pad_token_label_id]] * padding_length) + trigger_start_label_ids
            trigger_end_label_ids = ([[pad_token_label_id]] * padding_length) + trigger_end_label_ids
            role_start_label_ids = ([[pad_token_label_id]] * padding_length) + role_start_label_ids
            role_end_label_ids = ([[pad_token_label_id]] * padding_length) + role_end_label_ids
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            trigger_start_label_ids += [[pad_token_label_id]] * padding_length
            trigger_end_label_ids += [[pad_token_label_id]] * padding_length
            role_start_label_ids += [[pad_token_label_id]] * padding_length
            role_end_label_ids += [[pad_token_label_id]] * padding_length
        
        # print(len(label_ids), max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(trigger_start_label_ids) == max_seq_length
        assert len(trigger_end_label_ids) == max_seq_length
        assert len(role_start_label_ids) == max_seq_length
        assert len(role_end_label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s", example.id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))
            logger.info("trigger_start_label_ids: %s", " ".join([str(x) for x in trigger_start_label_ids]))
            logger.info("trigger_end_label_ids: %s", " ".join([str(x) for x in trigger_end_label_ids]))
            logger.info("role_start_label_ids: %s", " ".join([str(x) for x in role_start_label_ids]))
            logger.info("role_end_label_ids: %s", " ".join([str(x) for x in role_end_label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                trigger_start_label_ids=trigger_start_label_ids, trigger_end_label_ids= trigger_end_label_ids, \
                role_start_label_ids=role_start_label_ids, role_end_label_ids= role_end_label_ids )
        )
    return features

