#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Date    : 2020/3/24 10:12
 @Author  : LiHuaijun(lihuaijun@cmbchina.com) 
 @FileName: metrics.py
 @Version : beta
 @Desc: 现在计算过程中有对标注的事件做计数，统计后可以设成全局常量
"""
import json
from numpy import argmax
from absl import app, logging
import copy
import time


class IdError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def readjson(jsonfile):
    '''
    read json file and transformer to dict
    :param jsonfile:
    :return: idresmap
    '''
    idresmap = {}
    with open(jsonfile, "r", encoding="utf-8") as fw:
        for i in fw.readlines():
            linemap = json.loads(i)
            id = linemap["id"]
            idresmap[id] = linemap
    return idresmap


def gettypetrigger(event):
    '''
    get event type and trigger
    :param event:
    :return:
    '''
    typeword = event.get("type", "")
    triggerword = None
    for i in event.get("mentions", []):
        if i.get("role", "") == "trigger":
            triggerword = i.get("word", "")
            break
    # assert triggerword is not None, "There is no trigger"
    return typeword, triggerword


def mention_word_count(true_mentions, pred_mentions):
    '''
    按照word对比真实与预测的mention内容
    :param true_mentions:
    :param pred_mentions:
    :return: tp_int
    '''
    tp_int = 0
    pred_mentions_copy = copy.deepcopy(pred_mentions)
    for true_element in true_mentions:
        for ind, pred_element in enumerate(pred_mentions_copy):
            if true_element["role"] == pred_element["role"]:
                true_word = true_element["word"]
                pred_word = pred_element["word"]
                if true_word == pred_word:
                    tp_int += 1
                    pred_mentions_copy.pop(ind)
                break
    return tp_int


def mentionscount(true_mentions, pred_mentions):
    '''
    按照span对比真实与预测的mention内容
    :param true_mentions:
    :param pred_mentions:
    :return:
    '''
    tp_int = 0
    pred_mentions_copy = copy.deepcopy(pred_mentions)
    for true_element in true_mentions:
        for ind, pred_element in enumerate(pred_mentions_copy):
            if true_element["role"] == pred_element["role"]:
                true_begin, true_end = true_element["span"]
                pred_begin, pred_end = pred_element["span"]
                if true_begin == pred_begin and true_end == pred_end:
                    tp_int += 1
                    pred_mentions_copy.pop(ind)
                break
    return tp_int


def eventscore(true_event, pred_events):
    '''
    事件级别的得分计算，事件类型的判断不计入得分
    除去trigger一致的比对条件
    :param true_event:
    :param pred_events:
    :return:
    '''
    event_type, event_trigger = gettypetrigger(true_event)
    true_mentions = true_event["mentions"]
    elements_num = len(true_mentions)
    score_list = []
    match_trigger_index = []
    f1_list = []
    for i, pevent in enumerate(pred_events):
        score = 0.0
        recall = 0.0
        ptype, ptrigger = gettypetrigger(pevent)
        pred_mentions = pevent["mentions"]
        if event_type == ptype:  # and event_trigger == ptrigger
            match_trigger_index.append(i)
            # score += 1 / elements_num
            mention_num = mention_word_count(true_mentions, pred_mentions)
            score += mention_num / elements_num
            recall += mention_num / len(pred_mentions) if len(pred_mentions) != 0 else 0
            f1 = 2 * score * recall / (score + recall) if score + recall != 0.0 else 0
            score_list.append(score)
            f1_list.append(f1)
    if len(score_list) > 0:
        max_score = score_list[argmax(f1_list)]
        pred_events.pop(match_trigger_index[argmax(f1_list)])
    else:
        max_score = 0.0
    return max_score


def F1Score(truemap, predmap):
    '''
    计算F1 得分,若ID比对不上，抛出IdError异常
    :param truemap:
    :param predmap:
    :return: f1score
    '''
    true_event_num = 0
    pred_event_num = 0
    tp = 0.0
    for sentid, value in truemap.items():
        trueevent = value["events"]
        true_event_num += len(trueevent)
        pred_record = predmap.get(sentid, {})
        if len(pred_record) == 0:
            raise IdError("id:{id} is error, not in submission".format(id=sentid))
        else:
            predevents = pred_record.get("events", [])
            pred_event_num += len(predevents)
            for event in trueevent:
                try:
                    score = eventscore(true_event=event, pred_events=predevents)
                except Exception as e:
                    logging.error(e)
                    logging.error("error events is {id}".format(id=sentid))
                    score = 0.0
                tp += score
    precision = tp / pred_event_num if pred_event_num != 0 else 0
    recall = tp / true_event_num
    f1score = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0
    return f1score


def compute_metric(truefile,predfile):
    '''
    测试时的入口函数，确定脚本的输入和输出后，可作修改
    :param _:
    :return:
    '''
    truemap = readjson(truefile)
    predmap = readjson(predfile)
    ids = list(predmap.keys())
    for id in ids:
        if id not in truemap:
            predmap.pop(id)
            # print(id)
    
    starttime = time.time()
    f1score = F1Score(truemap, predmap)
    endtime = time.time()
    # print('spend time is {d}'.format(d=endtime - starttime))
    # print(f1score)
    return f1score


def compute_metric2(truefile, predfile):
    truemap = readjson(truefile)
    predmap = readjson(predfile)
    ids = list(predmap.keys())
    for id in ids:
        if id not in truemap:
            predmap.pop(id)
            print(id)
    role_pred, role_true = [], []
    trigger_pred, trigger_true = [], []
    for id, item in predmap.items():
        for event in item['events']:
            event_type = event['type']
            for mention in event['mentions']:
                if mention['role'] == 'trigger':
                    mention['span'][1] = mention['span'][0] + len(mention['word'])
                    trigger_pred.append([id] + mention['span'] + [event_type])
                else:
                    role_pred.append([id] + mention['span'] + [event_type + mention['role']])
    
    for id, item in truemap.items():
        for event in item['events']:
            event_type = event['type']
            for mention in event['mentions']:
                if mention['role'] == 'trigger':
                    trigger_true.append([id] + mention['span'] + [event_type])
                else:
                    role_true.append([id] + mention['span'] + [event_type + mention['role']])
    
    from .metrics import _precision_score, _recall_score, _f1_score
    trigger_result = [ _precision_score(trigger_true, trigger_pred), \
        _recall_score(trigger_true, trigger_pred), _f1_score(trigger_true, trigger_pred) ]

    role_result = [ _precision_score(role_true, role_pred), \
        _recall_score(role_true, role_pred), _f1_score(role_true, role_pred) ]
    print(trigger_result, role_result)
    return trigger_result, role_result


# def convert(input_file, outpu_file):
#     with open(input_file, "r", encoding="utf-8") as fw:
#         for i in fw.readlines():
#             linemap = json.loads(i)
#             for event in linemap['events']:
#                 event['']
#             results.append(linemap)



if __name__ == "__main__":
    truefile = "./data/FewFC-main/rearranged/test_trans.json"  # lhj-2361-2446mtest350dev0331_new
    # noisedfile = "noised.json"
    # predfile = "./result/test_trans_bin.json"
    predfile = "./output/ccks/trans/joint4/checkpoint-best/eval_predictions.json"
    metric = compute_metric2(truefile, predfile)
    print(metric)