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
""" GLUE processors and helpers """

import logging
import os
import json

from utils import get_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, id, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            id: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

## ccks格式
def event_type_process_ccks(input_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = []
        if is_predict: 
            results.append({"id":row["id"], "text_a":row["content"], "label":labels})
            continue
        for event in row["events"]:
            event_type = event["type"]
            labels.append(event_type)
        results.append({"id":row["id"], "text_a":row["content"],  "label":labels})
    # write_file(results,output_file)
    return results

## lic格式
def event_type_process_lic(input_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = []
        if is_predict: 
            results.append({"id":row["id"], "text_a":row["text"], "label":labels})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            labels.append(event_type)
        results.append({"id":row["id"], "text_a":row["text"], "label":labels})
    # write_file(results,output_file)
    return results

def read_examples_from_file(data_dir, mode, task, dataset="ccks"):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    if dataset=="ccks":
        items = event_type_process_ccks(file_path)
    elif dataset=="lic":
        items = event_type_process_lic(file_path)
    return [InputExample(**item) for item in items]


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
    mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_seq_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(len(attention_mask), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids), max_seq_length)

        label = [label_map[sub_label] for sub_label in  example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %s)" % (str(example.label), str(label)))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    return features


def convert_label_ids_to_onehot(label_ids, label_list):
    one_hot_labels= [0]*len(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}
    ignore_index= -100
    non_index= -1
    for label_id in label_ids:
        if label_id not in [ignore_index, non_index]:
            one_hot_labels[label_id]= 1
    return one_hot_labels