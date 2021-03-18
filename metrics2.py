
import json
from collections import defaultdict

def event_metric_f1(event_l, event_p):
    mentions_l = event_l['mentions']
    mentions_p = event_p['mentions']
    # 不考虑 span
    for mention in mentions_l:
        if 'span' in mentions_l:
            mention.pop('span') 
    for mention in mentions_p:
        if 'span' in mentions_p:
            mention.pop('span')
    
    tp = 0
    m = len(mentions_p)
    n = len(mentions_l)
    for mention in mentions_l:
        if mention in mentions_p:
            tp += 1
    if m==0 and n==0: return 1
    elif m==0 or n==0: return 0
    elif tp==0: return 0

    pr = tp/m
    re = tp/n
    f1 = 2*pr*re/(pr+re)
    return f1

def event_metric_recall(event_l, event_p):
    mentions_l = event_l['mentions']
    mentions_p = event_p['mentions']
    # 不考虑 span
    for mention in mentions_l:
        if 'span' in mentions_l:
            mention.pop('span') 
    for mention in mentions_p:
        if 'span' in mentions_p:
            mention.pop('span')
    
    score = 0
    k = len(mentions_l)
    for mention in mentions_l:
        if mention in mentions_p:
            score += 1/k
    return score


def compute_matric(label_file, pred_file):
    label_lines = open(label_file, encoding='utf-8').read().splitlines()
    pred_lines = open(pred_file, encoding='utf-8').read().splitlines()
    
    labels = {}
    for i, line in enumerate(label_lines):
        json_line = json.loads(line)
        id = json_line['id']
        events = json_line['events']
        labels[id] = events
        # n += len(events)

    preds = {}
    for i, line in enumerate(pred_lines):
        json_line = json.loads(line)
        id = json_line['id']
        events = json_line['events']
        preds[id] = events
        # m += len(events)

    # assert len(labels) == len(preds)

    m, n, tp = 0, 0, 0
    for id, events_l in labels.items():
        events_p  = preds[id]
        m += len(events_p)
        n += len(events_l)
        type_list = list(set([event['type'] for event in events_l]))
        candidates = defaultdict(list)
        for i, event_l in enumerate(events_l):
            for j, event_p in enumerate(events_p):
                if event_l['type'] == event_p['type']:
                    type = event_l['type']
                    candidates[type].append([i, j, event_metric_recall(event_l, event_p)])
        
        for type in type_list:
            candidates[type].sort(key=lambda x:x[-1], reverse=True)
            i_ignore = []
            j_ignore = []
            # while candidates[type]:
            for i, j, score in candidates[type]:
                if i not in i_ignore and j not in j_ignore:
                    tp += score
                    i_ignore.append(i)
                    j_ignore.append(j)
    pr = tp/m
    re = tp/n
    f1 = 2*pr*re/(pr+re)
    print(pr, re, f1)
    return f1

metric = compute_matric("./data/FewFC-main/rearranged/test_trans.json",
    "./result/test_trans.json")
print(metric)


