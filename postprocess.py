#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""hello world"""

import json
from seqeval.metrics.sequence_labeling import get_entities

from utils import get_labels, write_file

def read_by_lines(path, encoding="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w") as outfile:
        [outfile.write(d + "\n") for d in data]

# def _extract_entities(text, labels):
#     """extract_entities"""
#     ret, is_start, cur_type = [], False, None
#     for i, label in enumerate(labels):
#         if label != u"O":
#             _type = label[2:]
#             if label.startswith(u"B-"):
#                 is_start = True
#                 cur_type = _type
#                 ret.append({"start": i, "text": [text[i]], "type": _type})
#             elif _type != cur_type:
#                 """
#                 # 如果是没有B-开头的，则不要这部分数据
#                 cur_type = None
#                 is_start = False
#                 """
#                 cur_type = _type
#                 is_start = True
#                 ret.append({"start": i, "text": [text[i]], "type": _type})
#             elif is_start:
#                 ret[-1]["text"].append(text[i])
#             else:
#                 cur_type = None
#                 is_start = False
#         else:
#             cur_type = None
#             is_start = False

#     for item in ret:
#         item['text']= ''.join(item['text'])
#     return ret

def extract_entities(text, labels):
    items = get_entities(labels)
    ret = []
    for type, i, j in items:
        ret.append([i, text[i:j+1], type])
    return ret


## trigger-bio + role-bio: 输出为lic2020文件形式
def predict_data_process_bio2(test_file, trigger_file, role_file, schema_file, save_path):
    """predict_data_process_bio"""
    pred_ret = []
    test_datas = read_by_lines(test_file)
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        s = json.loads(s)
        schema[s["event_type"]] = [r["role"] for r in s["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for test_data, role_data in zip(test_datas, role_datas):
        # arguments = json.loads(data)["arguments"]
        test_data = json.loads(test_data)
        text = test_data['text']
        labels = json.loads(role_data)['labels']
        arguments = extract_entities(text, labels)
        sent_role_mapping[test_data["id"]] = arguments

    for test_data, trigger_data in zip(test_datas, trigger_datas):
        # arguments = json.loads(data)["arguments"]
        test_data = json.loads(test_data)
        text = test_data['text']
        labels = json.loads(trigger_data)['labels']
        triggers = extract_entities(text, labels)
        event_list = []
        for trigger_start_index, trigger, event_type in triggers:
            role_list = schema[event_type]
            arguments = []
            for arg_start_index, arg, role_type in sent_role_mapping[test_data["id"]]:
                if role_type not in role_list:
                    continue
                if len(arg) == 1: continue
                arguments.append({"role": role_type, "argument_start_index":arg_start_index, "argument": arg})
            event = {"event_type": event_type, "trigger_start_index":trigger_start_index, "trigger":trigger, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": test_data["id"],
            "text": text,
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

## trigger-bio + role-bio: 输出为ccks2020文件形式
def predict_data_process_bio(test_file, trigger_file, role_file, schema_file, save_path):
    """predict_data_process_bio"""
    pred_ret = []
    test_datas = read_by_lines(test_file)
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        s = json.loads(s)
        schema[s["event_type"]] = [r["role"] for r in s["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for test_data, role_data in zip(test_datas, role_datas):
        # arguments = json.loads(data)["arguments"]
        test_data = json.loads(test_data)
        text = test_data['content']
        labels = json.loads(role_data)['labels']
        arguments = extract_entities(text, labels)
        sent_role_mapping[test_data["id"]] = arguments

    for test_data, trigger_data in zip(test_datas, trigger_datas):
        # arguments = json.loads(data)["arguments"]
        test_data = json.loads(test_data)
        text = test_data['content']
        labels = json.loads(trigger_data)['labels']
        triggers = extract_entities(text, labels)
        event_list = []
        for trigger_start_index, trigger, event_type in triggers:
            role_list = schema[event_type]
            arguments = [{"role": "trigger","span":[trigger_start_index, trigger_start_index+ len(trigger)], "word": trigger}]
            for arg_start_index, arg, role_type in sent_role_mapping[test_data["id"]]:
                if role_type not in role_list:
                    continue
                if len(arg) == 1: continue
                arguments.append({"role": role_type, "span":[arg_start_index, arg_start_index+ len(arg)], "word": arg})
            event = {"type": event_type, "mentions": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": test_data["id"],
            "content": text,
            "events": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

def ensemble(input_file, output_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    res =[]
    pre_line = {"id":""}
    flag= False
    for line in lines:
        json_line = json.loads(line)
        cur_id = json_line["id"]
        pre_id = pre_line["id"]
        if cur_id != pre_id:
            res.append(json_line)
            pre_id = cur_id
            pre_line = json_line
            flag= True
        else:
            json_line["event_list"].extend(pre_line["event_list"])
            pre_line = json_line
            flag= False
    if not flag:
        res.append(json_line)
    
    from preprocess import write_file
    write_file(res, output_file)


if __name__ == "__main__":
    predict_data_process_bio(
        test_file = "./data/FewFC-main/rearranged/trans/0/test.json",
        trigger_file= "./output/trigger_trans2/0/checkpoint-best/test_predictions.json", \
        role_file = "./output/role_trans2/0/checkpoint-best/test_predictions.json", \
        schema_file = "./data/event_schema.json", \
        save_path =  "./result/test_trans2.json")

    # predict_data_process_bin(
    #     trigger_file= "./output/trigger_classify/0/checkpoint-best/test_predictions_indexed.json", \
    #     role_file = "./output/role_bin/0/checkpoint-best/test_predictions_indexed.json", \
    #     schema_file = "./data/ccks4_2/event_schema.json", \
    #     save_path =  "./results/test_pred_trigger_bin_role_bin_split.json")

    # ensemble("./output/role_segment_bin/checkpoint-best/eval_predictions_indexed.json",\
    #       "./results/eval_pred_bi_segment.json")
    # ensemble("./output/role_segment_bin/checkpoint-best/test_predictions_indexed.json",\
    #       "./results/test_pred_bi_segment.json")

