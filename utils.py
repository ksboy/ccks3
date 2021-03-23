import json
import re

from transformers import BertTokenizer
class OurBertTokenizer(BertTokenizer):
    def _tokenize(self, text):
        R = ["[CLS]"]
        for c in text:
            if c in self.vocab:
                R.append(c)
            elif c == ' ':
                R.append('[unused1]')
            # elif c == '“' or c == '”':
            #     R.append('"')
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        R.append("[SEP]")
        return R

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")

def remove_duplication(alist):
    res = []
    for item in alist:
        if item not in res:
            res.append(item)
    return res


def get_labels(path="./data/event_schema.json", task='trigger', mode="ner"):
    if not path:
        if mode=='ner':
            return ["O", "B-ENTITY", "I-ENTITY"]
        else:
            return ["O"]

    elif task=='trigger':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            event_type = row["event_type"]
            if mode == "ner":
                labels.append("B-{}".format(event_type))
                labels.append("I-{}".format(event_type))
            else:
                labels.append(event_type)
        return remove_duplication(labels)

    elif task=='role':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            for role in row["role_list"]:
                role_type = role['role']
                if mode == "ner":
                    labels.append("B-{}".format(role_type))
                    labels.append("I-{}".format(role_type))
                else:
                    labels.append(role_type)
        return remove_duplication(labels)
    # 特定类型事件 [TASK] 中的角色
    else:
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            if row['class']!=task:
                continue
            for role in row["role_list"]:
                role_type = role['role']
                if mode == "ner":
                    labels.append("B-{}".format(role_type))
                    labels.append("I-{}".format(role_type))
                else:
                    labels.append(role_type)
        return remove_duplication(labels)


def find_all(a_str, sub):
    start = 0
    results = []
    while True:
        start = a_str.find(sub, start)
        if start == -1: break
        results.append(start)
        start += 1 # use start += 1 to find overlapping matches
    return results

def _split(review, pattern):
    split_index_list = []
    pre_split_index= 0
    pre_index = 0
    for m in re.finditer(pattern, review):
        split_index = m.span()[1]
        if split_index - pre_split_index > 510 -1:
            split_index_list.append(pre_index)
            pre_split_index = pre_index
        pre_index = split_index
    if len(review) - pre_split_index > 510 -1 and "split_index" in dir() :
        split_index_list.append(split_index)
    split_index_list = [0] + split_index_list + [10000000]
    sub_reviews = []
    for i in range(len(split_index_list)-1):
        sub_reviews.append(review[split_index_list[i]:split_index_list[i+1]])
    while "" in sub_reviews:
        sub_reviews.remove("")
    return sub_reviews

def get_sub(text):
    patterns =  ["。", "；","，","、"]
    reviews = [text]
    len_max_reviews = len(reviews[0])
    for pattern in patterns:
        if len_max_reviews > 510:
            _reviews =[]
            for review in reviews:
                if len(review) > 510 :
                    _reviews += _split(review, pattern)
                else:
                    _reviews.append(review)
            reviews = _reviews
            len_max_reviews = max([len(review) for review in reviews])
        else:
            break
    return reviews


def get_num_of_arguments(input_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    arg_count = 0
    for line in lines:
        line = json.loads(line)
        for event in line["event_list"]:
            arg_count += len(event["arguments"])
    print(arg_count)

def read_write(input_file, output_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        row = json.loads(row)
        id = row.pop('id')
        text = row.pop('text')
        # labels = row.pop('labels')
        event_list = row.pop('event_list')
        row['text'] = text
        row['id'] = id
        # row['labels'] = labels
        row['event_list'] = event_list
        results.append(row)
    write_file(results, output_file)

if __name__ == '__main__':
    # labels = get_labels(path="./data/event_schema/base.json", task='role', mode="classification")
    # print(len(labels))
    

    # get_num_of_arguments("./results/test_pred_bin_segment.json")

    # read_write("./output/eval_pred.json", "./results/eval_pred.json")
    # read_write("./results/test1.trigger.pred.json", "./results/paddle.trigger.json")
    pass
