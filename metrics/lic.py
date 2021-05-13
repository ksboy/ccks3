
from metrics import _f1_score
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

