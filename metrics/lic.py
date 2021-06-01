
from .metrics import _f1_score
def token_level_metric(label_list, preds_list):
    """
    统计所有论元字符级别的PRF值。首先需要计算每个论元的PRF,而且注意label_list的每行中可能包含多个论元需要单独计算
    :param label_list: [["xxx", "xx", ...]*data_nums]，内层列表中是当前事件类型和论元角色下的所有论元字符串
    :param preds_list: [["xxx", "xx", ...]*data_nums]
    :return: token_level_precision, token_level_recall, token_level_f1
    """
    all_label_roles_num, all_pred_roles_num = 0, 0
    all_pred_role_score = 0

    for i in range(len(label_list)):
        all_label_roles_num += len(label_list[i])
    for i in range(len(preds_list)):
        all_pred_roles_num += len(preds_list[i])
    for i in range(len(label_list)):
        pred_labels = preds_list[i][:]
        for _label in label_list[i]:
            _f1 = [_f1_score(_label, _pred) for _pred in pred_labels]
            all_pred_role_score += max(_f1) if _f1 else 0

    token_level_precision = all_pred_role_score / all_pred_roles_num if all_pred_roles_num else 0
    token_level_recall = all_pred_role_score / all_label_roles_num if all_label_roles_num else 0
    token_level_f1 = 2 * token_level_precision * token_level_recall / (token_level_precision + token_level_recall) if token_level_precision + token_level_recall else 0

    return token_level_precision, token_level_recall, token_level_f1

from .ccks3 import readjson
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
                    # mention['span'][1] = mention['span'][0] + len(mention['word'])
                    trigger_pred.append([id] + mention['span'] + [event_type])
                else:
                    role_pred.append([id] + mention['span'] + [event_type + mention['role']])
    
    for id, item in truemap.items():
        for event in item['event_list']:
            event_type = event['event_type']
            trigger = event['trigger']
            trigger_start_index = event['trigger_start_index']
            trigger_end_index = trigger_start_index + len(trigger)
            trigger_true.append([id] + [trigger_start_index, trigger_end_index] + [event_type])
            for argument in event['arguments']:
                argument_start_index = argument['argument_start_index']
                argument_end_index = argument_start_index + len(argument)
                role_true.append([id] + [argument_start_index, argument_end_index] + [event_type + argument['role']])
    
    from .metrics import _precision_score, _recall_score, _f1_score
    trigger_result = [ _precision_score(trigger_true, trigger_pred), \
        _recall_score(trigger_true, trigger_pred), _f1_score(trigger_true, trigger_pred) ]

    role_result = [ _precision_score(role_true, role_pred), \
        _recall_score(role_true, role_pred), _f1_score(role_true, role_pred) ]
    print(trigger_result, role_result)
    return trigger_result, role_result
