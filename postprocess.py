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
import os
import sys
import json
import argparse
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


def index_output_bio_trigger(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        tokens = test.pop('tokens')
        test['text'] = ''.join(tokens)
        if "labels" in test: test.pop("labels")

        prediction = json.loads(prediction)
        labels = prediction["labels"]
        if len(labels)!=len(tokens) and len(labels) != max_length:
            print(labels, tokens)
            print(len(labels), len(tokens), index)
            break
        t_ret = extract_result(test['text'], labels)
        test["triggers"] = t_ret

        results.append(test)
    write_file(results, output_file)

def index_output_bin_trigger(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)

        prediction = json.loads(prediction)
        labels = prediction["labels"]
        test["labels"] = labels

        results.append(test)
    write_file(results, output_file)

def index_output_bio_arg(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        tokens = test.pop('tokens')
        test['text'] = ''.join(tokens)

        prediction = json.loads(prediction)
        labels = prediction["labels"]
        if len(labels)!=len(tokens) and len(labels) != max_length:
            print(labels, tokens)
            print(len(labels), len(tokens), index)
            break

        args = extract_result(test["text"], labels)
        arguments = []
        for arg in args:
            argument = {}
            argument["role"] = arg["type"]
            argument["argument_start_index"] = arg['start']
            argument["argument"] =''.join(arg['text'])
            arguments.append(argument)
        
        test.pop("labels")
        test["arguments"] = arguments
        results.append(test)
    write_file(results, output_file)


def index_output_segment_bin(test_file, prediction_file, output_file):
    label_list = get_labels(task='role', mode="classification")
    label_map =  {i: label for i, label in enumerate(label_list)}

    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        start_labels = test.pop('start_labels')
        end_labels = test.pop('end_labels')

        tokens = test.pop('tokens')
        text = ''.join(tokens)
        test['text'] = text

        segment_ids =  test.pop('segment_ids')
        trigger = ''.join([tokens[i] for i in range(len(tokens)) if segment_ids[i]])
        for i in range(len(tokens)):
            if segment_ids[i]:
                trigger_start_index = i
                break
        
        event = {}
        # event['trigger'] = trigger
        # event['trigger_start_index']= trigger_start_index
        event_type = test.pop("event_type")
        event["event_type"]=event_type

        prediction = json.loads(prediction)
        arg_list = prediction["labels"]
        arguments =[]
        for arg in arg_list:
            sub_dict = {}
            argument_start_index = arg[1] -1 
            argument_end_index = arg[2] -1 
            argument = text[argument_start_index:argument_end_index+1]
            role = label_map[arg[3]]
            sub_dict["role"]=role
            sub_dict["argument"]=argument
            # sub_dict["argument_start_index"] = argument_start_index
            arguments.append(sub_dict)
        
        event["arguments"]= arguments

        test['event_list']= [event]
        results.append(test)
    write_file(results, output_file)

def index_output_bin_arg(test_file, prediction_file, output_file):
    label_list = get_labels(task='role', mode="classification")
    label_map =  {i: label for i, label in enumerate(label_list)}

    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        start_labels = test.pop('start_labels')
        end_labels = test.pop('end_labels')

        tokens = test.pop('tokens')
        text = ''.join(tokens)
        test['text'] = text

        prediction = json.loads(prediction)
        arg_list = prediction["labels"]
        arguments =[]
        for arg in arg_list:
            sub_dict = {}
            argument_start_index = arg[1] -1 
            argument_end_index = arg[2] -1 
            argument = text[argument_start_index:argument_end_index+1]
            role = label_map[arg[3]]
            sub_dict["role"]=role
            sub_dict["argument"]=argument
            sub_dict["argument_start_index"] = argument_start_index
            arguments.append(sub_dict)
        
        test["arguments"]= arguments
        results.append(test)
    write_file(results, output_file)


# un-finished
# def binary_to_bio(test_file, prediction_file, output_file):
#     tests = open(test_file, encoding='utf-8').read().splitlines()
#     predictions = open(prediction_file, encoding='utf-8').read().splitlines()
#     results = []
#     for test,prediction in zip(tests, predictions):
#         test = json.loads(test)
#         tokens = test.pop('tokens')
#         test['text'] = ''.join(tokens)

#         row_preds_list = json.loads(prediction)
        
#         labels= ['O']*len(tokens)
#         # for pred in row_preds_list:
        
#         test.update(prediction) 

#         results.append(test)
#     write_file(results, output_file)

# ner_segment_bi 输入的预处理函数

def convert_bio_to_segment(input_file, output_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    res = []
    for line in lines:
        line = json.loads(line)
        text = line["text"]
        labels = line["labels"]
        tokens = list(text)
        if len(labels)!=len(tokens):
            print(len(labels), len(tokens))

        triggers = extract_result(text, labels)
        if len(triggers)==0:
            print("detect no trigger")
        for trigger in triggers:
            event_type= trigger["type"]
            segment_ids = [0]*(len(tokens))
            trigger_start_index = trigger['start']
            trigger_end_index = trigger['start'] + len(trigger['text'])
            for i in range(trigger_start_index, trigger_end_index):
                segment_ids[i] = 1
            start_labels = ['O']*(len(tokens))
            end_labels =  ['O']*(len(tokens))

            cur_line = {}
            cur_line["id"] = line["id"]
            cur_line["tokens"] = tokens
            cur_line["event_type"] = event_type
            cur_line["segment_ids"] = segment_ids
            cur_line["start_labels"] = start_labels
            cur_line["end_labels"] = end_labels
            res.append(cur_line)
    write_file(res, output_file)


def convert_bio_to_label(input_file, output_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    res = []
    for line in lines:
        line_json = json.loads(line)
        labels = []
        for label in line_json["labels"]:
            if label.startswith("B-") and label[2:] not in labels:
                labels.append(label[2:])
        res.append({"labels":labels})
    write_file(res, output_file)

def compute_matric(label_file, pred_file):
    label_lines = open(label_file, encoding='utf-8').read().splitlines()
    pred_lines = open(pred_file, encoding='utf-8').read().splitlines()

    labels = []
    for i, line in enumerate(label_lines):
        json_line = json.loads(line)
        for label in json_line['labels']:
            labels.append([i, label])
    
    preds = []
    for i, line in enumerate(pred_lines):
        json_line = json.loads(line)
        for label in json_line['labels']:
            preds.append([i, label])

    nb_correct  = 0
    for out_label in labels:
        if out_label in preds:
            nb_correct += 1
            continue
    nb_pred = len(preds)
    nb_true = len(labels)
    # print(nb_correct, nb_pred, nb_true)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
    print(p, r, f1)

def data_process(path, model="trigger", is_predict=False):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = u"B-" if i == start else u"I-"
            data[i] = u"{}{}".format(suffix, _type)
        return data

    sentences = []
    output = [u"text_a"] if is_predict else [u"text_a\tlabel"]
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip().decode("utf-8"))
            _id = d_json["id"]
            text_a = [
                u"，" if t == u" " or t == u"\n" or t == u"\t" else t
                for t in list(d_json["text"].lower())
            ]
            if is_predict:
                sentences.append({"text": d_json["text"], "id": _id})
                output.append(u'\002'.join(text_a))
            else:
                if model == u"trigger":
                    labels = [u"O"] * len(text_a)
                    for event in d_json["event_list"]:
                        event_type = event["event_type"]
                        start = event["trigger_start_index"]
                        trigger = event["trigger"]
                        labels = label_data(labels, start,
                                            len(trigger), event_type)
                    output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                   u'\002'.join(labels)))
                elif model == u"role":
                    for event in d_json["event_list"]:
                        labels = [u"O"] * len(text_a)
                        for arg in event["arguments"]:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            start = arg["argument_start_index"]
                            labels = label_data(labels, start,
                                                len(argument), role_type)
                        output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                       u'\002'.join(labels)))
    if is_predict:
        return sentences, output
    else:
        return output


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if u"B-{}".format(_type) not in labels:
            labels.extend([u"B-{}".format(_type), u"I-{}".format(_type)])
        return labels

    labels = []
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip().decode("utf-8"))
            if model == u"trigger":
                labels = label_add(labels, d_json["event_type"])
            elif model == u"role":
                for role in d_json["role_list"]:
                    labels = label_add(labels, role["role"])
    labels.append(u"O")
    return labels


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False

    for item in ret:
        item['text']= ''.join(item['text'])
    return ret

## trigger-ner + role-ner
def predict_data_process_ner(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["text"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

## trigger-ner + role-bin
def predict_data_process_ner_bin(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        arguments =d_json["arguments"]
        role_ret = {}
        for r in arguments:
            role_type = r["role"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append([u"".join(r["argument"]),r["argument_start_index"]])
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = d_json["triggers"]
        pred_event_types = [[t["type"], t["text"], t["start"] ] for t in t_ret]
        event_list = []
        for event_type, trigger, trigger_start_index in pred_event_types:
            if len(trigger) == 1:
                # 一点小trick
                continue
            role_list = schema[event_type]
            arguments = [{"role": "trigger","span":[trigger_start_index, trigger_start_index+ len(trigger)], "word": trigger}]
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg, arg_start_index in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments.append({"role": role_type, "span":[arg_start_index, arg_start_index+ len(arg)], "word": arg})
            event = {"type": event_type, "mentions": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "events": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

## trigger-bin + role-bin
def predict_data_process_bin(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    schema_reverse = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
        schema_reverse=[]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        arguments =d_json["arguments"]
        role_ret = {}
        for r in arguments:
            role_type = r["role"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["argument"]))
        if d_json["id"] not in sent_role_mapping:
            sent_role_mapping[d_json["id"]] = {}
        sent_role_mapping[d_json["id"]].update(role_ret)

    for d in trigger_datas:
        d_json = json.loads(d)
        pred_event_types = d_json["labels"]
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = {}
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments[role_type] = arg
            event = {"event_type": event_type}
            event.update(arguments)
            event_list.append(event)
        pred_ret.append({
            "doc_id": d_json["id"],
            "events": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

## trigger-bin + role-bin + 以role为准
def predict_data_process_bin2(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    from utils import schema_analysis
    argument_map= schema_analysis()

    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    schema_reverse = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
        schema_reverse=[]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        arguments =d_json["arguments"]
        role_ret = {}
        for r in arguments:
            role_type = r["role"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["argument"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        pred_event_types = d_json["labels"]
        event_list = []
        
        for role_type, ags in sent_role_mapping[d_json["id"]].items():
            for event_type in pred_event_types:
                role_list = schema[event_type]
                if role_type not in role_list:
                    event_type = argument_map
                    pred_event_types.append
                    continue
                for arg in ags:
                    # 一点小trick
                    if len(arg) == 1: continue
                    arguments.append({"role": role_type, "argument": arg})
            

            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)


        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
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


# def merge(label_file, ):


if __name__ == "__main__":

    # index_output_bio_trigger("./data/trigger_base/0/dev.json" , "./output/trigger_base/0/checkpoint-best/eval_predictions.json","./output/trigger_base/0/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bio_trigger("./data/trigger_base/test.json" , "./output/trigger_base/0/checkpoint-best/test_predictions.json","./output/trigger_base/0/checkpoint-best/test_predictions_indexed.json" )
    # index_output_bio_trigger("./data/trigger_trans/0/dev.json" , "./output/trigger_base/0/checkpoint-best/eval_predictions.json","./output/trigger_base/0/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bio_trigger("./data/trigger_trans/test.json" , "./output/trigger_all/0/checkpoint-best/test_predictions.json","./output/trigger_all/0/checkpoint-best/test_predictions_indexed.json" )
    index_output_bio_trigger("./data/FewFC-main/trigger_trans/test.json" , "./output/trigger_trans/0/checkpoint-best/test_predictions.json","./output/trigger_trans/0/checkpoint-best/test_predictions_indexed.json" )

    
    # index_output_bin_trigger("./data/trigger_classify/dev.json" , "./output/trigger_classify/merge/eval_predictions_labels.json","./output/trigger_classify/merge/eval_predictions_indexed_labels.json" )
    # index_output_bin_trigger("./data/trigger_classify/test.json" , "./output/trigger_classify/0/checkpoint-best/test_predictions.json","./output/trigger_classify/0/checkpoint-best/test_predictions_indexed.json" )

    # index_output_bio_arg("./data/role/dev.json" , "./output/role/checkpoint-best/eval_predictions.json","./output/role/checkpoint-best/eval_predictions_labels.json" )
    # index_output_bio_arg("./data/role/test.json" , "./output/role/checkpoint-best/test_predictions.json","./output/role/checkpoint-best/test_predictions_indexed.json" )

    # index_output_segment_bin("./data/role_segment_bin/dev.json" , "./output/role_segment_bin/checkpoint-best/eval_predictions.json","./output/role_segment_bin/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_segment_bin("./data/role_segment_bin/test.json" , "./output/role_segment_bin/checkpoint-best/test_predictions.json","./output/role_segment_bin/checkpoint-best/test_predictions_indexed.json" )

    # index_output_bin_arg("./data/role_base/0/dev.json" , "./output/role_base/0/checkpoint-best/eval_predictions.json","./output/role_base/0/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bin_arg("./data/role_base/test.json" , "./output/role_base/0/checkpoint-best/test_predictions.json","./output/role_base/0/checkpoint-best/test_predictions_indexed.json" )
    # index_output_bin_arg("./data/role_trans/0/dev.json" , "./output/role_base/0/checkpoint-best/eval_predictions.json","./output/role_base/0/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bin_arg("./data/role_trans/test.json" , "./output/role_trans2/checkpoint-best/test_predictions.json","./output/role_trans2/checkpoint-best/test_predictions_indexed.json" )
    # index_output_bin_arg("./data/role_all/0/dev.json" , "./output/role_all/0/checkpoint-best/eval_predictions.json","./output/role_all/0/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bin_arg("./data/role_trans/test.json" , "./output/role_all/0/checkpoint-best/test_predictions.json","./output/role_all/0/checkpoint-best/test_predictions_indexed.json" )

    # convert_bio_to_segment("./output/trigger/checkpoint-best/test_predictions_indexed.json",\
    #     "./output/trigger/checkpoint-best/test_predictions_indexed_semgent_id.json")

    # convert_bio_to_label("./output/trigger/checkpoint-best/eval_predictions.json",\
    #      "./output/trigger/checkpoint-best/eval_predictions_labels.json")
    # compute_matric("./data/trigger_classify/dev.json", "./output/trigger/checkpoint-best/eval_predictions_labels.json")


    # predict_data_process_ner(
    #     trigger_file= "./output/trigger/checkpoint-best/test_predictions_indexed.json", \
    #     role_file = "./output/role2/checkpoint-best/test_predictions_indexed.json", \
    #     schema_file = "./data/event_schema/event_schema.json", \
    #     save_path =  "./results/test_pred2.json")

    # predict_data_process_ner_bin(
    #     trigger_file= "./output/trigger_base/0/checkpoint-best/eval_predictions_indexed.json", \
    #     role_file = "./output/role_base/0/checkpoint-best/eval_predictions_indexed.json", \
    #     schema_file = "./data/event_schema.json", \
    #     save_path =  "./results/eval_base.json")
    # predict_data_process_ner_bin(
    #     trigger_file= "./output/trigger_base/0/checkpoint-best/test_predictions_indexed.json", \
    #     role_file = "./output/role_base/0/checkpoint-best/test_predictions_indexed.json", \
    #     schema_file = "./data/event_schema.json", \
    #     save_path =  "./results/test_base.json")
    # predict_data_process_ner_bin(
    #     trigger_file= "./output/trigger_all/0/checkpoint-best/test_predictions_indexed.json", \
    #     role_file = "./output/role_all/0/checkpoint-best/test_predictions_indexed.json", \
    #     schema_file = "./data/event_schema.json", \
    #     save_path =  "./results/test_trans2.json")
    # predict_data_process_ner_bin(
    #     trigger_file= "./output/trigger_trans/checkpoint-best/test_predictions_indexed.json", \
    #     role_file = "./output/role_trans2/checkpoint-best/test_predictions_indexed.json", \
    #     schema_file = "./data/event_schema copy.json", \
    #     save_path =  "./results/test_trans5.json")

    # predict_data_process_bin(
    #     trigger_file= "./output/trigger_classify/0/checkpoint-best/test_predictions_indexed.json", \
    #     role_file = "./output/role_bin/0/checkpoint-best/test_predictions_indexed.json", \
    #     schema_file = "./data/ccks4_2/event_schema.json", \
    #     save_path =  "./results/test_pred_trigger_bin_role_bin_split.json")

    # ensemble("./output/role_segment_bin/checkpoint-best/eval_predictions_indexed.json",\
    #       "./results/eval_pred_bi_segment.json")
    # ensemble("./output/role_segment_bin/checkpoint-best/test_predictions_indexed.json",\
    #       "./results/test_pred_bi_segment.json")

