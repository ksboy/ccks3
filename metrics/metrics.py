def precision_score(batch_labels, batch_preds):
    assert len(batch_labels) == len(batch_preds)
    nb_correct, nb_pred = 0, 0
    for labels, preds in zip(batch_labels, batch_preds):
        for label in labels:
            if label in preds:
                nb_correct += 1
        nb_pred += len(preds)
    p = nb_correct / nb_pred if nb_pred > 0 else 0
    return p

def recall_score(batch_labels, batch_preds):
    assert len(batch_labels) == len(batch_preds)
    nb_correct, nb_true = 0, 0
    for labels, preds in zip(batch_labels, batch_preds):
        for label in labels:
            if label in preds:
                nb_correct += 1
        nb_true += len(labels)
    r = nb_correct / nb_true if nb_true > 0 else 0
    return r

def f1_score(batch_labels, batch_preds):
    assert len(batch_labels) == len(batch_preds)
    nb_correct, nb_true, nb_pred = 0, 0, 0
    for labels, preds in zip(batch_labels, batch_preds):
        for label in labels:
            if label in preds:
                nb_correct += 1
        nb_true += len(labels)
        nb_pred += len(preds)
    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1

# 一维
def _precision_score(labels, preds):
    nb_correct = 0
    for label in labels:
        if label in preds:
            nb_correct += 1
            # continue
    nb_pred = len(preds)
    p = nb_correct / nb_pred if nb_pred > 0 else 0
    return p
    
# 一维
def _recall_score(labels, preds):
    nb_correct = 0
    for label in labels:
        if label in preds:
            nb_correct += 1
            # continue
    nb_true = len(labels)
    r = nb_correct / nb_true if nb_true > 0 else 0
    return r

# 一维
def _f1_score(labels, preds):
    nb_correct = 0
    for label in labels:
        if label in preds:
            nb_correct += 1
            # continue
    nb_pred = len(labels)
    nb_true = len(preds)
    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1
