import json
import re

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


def get_labels(path="./data/ccks4_2/event_schema.json", task='trigger', mode="ner"):
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

def data_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()

    event_type_count = 0
    event_type_count2 = 0
    role_count = 0
    arg_count = 0
    arg_count2 = 0
    arg_role_count = 0
    arg_role_one_event_count = 0
    # trigger_count = 0 # triggr

    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)

        arg_start_index_list=[]
        arg_start_index_map={}
        event_type_list = []
        # trigger_start_index_list = []

        event_type_flag = False  # 两个不同类型的事件
        event_type_flag2 = False # 两个事件
        arg_start_index_flag = False # 一个事件，一个 role 对应多个 arg: 相同/不同
        arg_start_index_flag2 = False # 一个事件，一个 role 对应多个 arg：不同
        role_flag = False # 一个/多个事件，一个arg对应多个role：相同/不同
        arg_role_flag= False # 一个/多个事件，一个arg对应多个role：不同
        arg_role_one_event_flag= False # 一个事件 一个arg对应多个role：不同
        # trigger_flag = False

        for event in row["event_list"]:
            event_type = event["event_type"]
            if event_type_list==[]: 
                event_type_list.append(event_type)
            else:
                event_type_flag2 = True
                print(row)
                if event_type not in event_type_list:
                    # event_class_count += 1
                    event_type_flag = True
                    # print(row)
            
            # trigger_start_index= event["trigger_start_index"]
            # if trigger_start_index not in trigger_start_index_list:
            #     trigger_start_index_list.append(trigger_start_index)
            # else:
            #     trigger_flag = True
            #     print(row)

            role_list = []
            role_map = {}
            arg_start_index_map_in_one_event = {}
            for arg in  event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                if role not in role_list:
                    role_list.append(role)
                    role_map[role] = argument
                else: 
                    arg_start_index_flag = True
                    if argument!= role_map[role]:
                        arg_start_index_flag2 = True
                        # print(row)
                
                if argument_start_index not in arg_start_index_map_in_one_event:
                    arg_start_index_map_in_one_event[argument_start_index]= role
                else:
                    if role!= arg_start_index_map_in_one_event[argument_start_index]:
                        arg_role_one_event_flag = True
                        # print(row)
                        # return 0


                if argument_start_index not in arg_start_index_list:
                    arg_start_index_list.append(argument_start_index)
                    arg_start_index_map[argument_start_index]= role
                else: 
                    role_flag = True
                    # print(row)
                    if role!= arg_start_index_map[argument_start_index]:
                        arg_role_flag = True
                        # print(row)
    
        if role_flag:
            role_count += 1
            # print(row)
        if event_type_flag:
            event_type_count += 1
            # print(row)
        if event_type_flag2:
            event_type_count2 += 1
            # print(row)
        if arg_start_index_flag:
            arg_count += 1
            # print(row)
        if arg_start_index_flag2:
            arg_count2 += 1
            # print(row)
        if arg_role_flag:
            arg_role_count += 1
        if arg_role_one_event_flag:
            arg_role_one_event_count += 1
        # if trigger_flag:
        #     trigger_count += 1
    
    print(event_type_count,event_type_count2, role_count, arg_count, arg_count2, arg_role_count, arg_role_one_event_count)



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


def convert(input_file, output_file, mode="train"):
    res = []
    rows = open(input_file, encoding='utf-8').read().splitlines()
    text_all_count = 0
    text_out_count = 0
    arg_all_count = 0
    arg_out_count = 0
    for row in rows:
        row = json.loads(row)
        # assert(sum([len(text) for text in _split(row['content'])])==len(row['content']))
        sub_text_list= get_sub(row['content'])
        for text in sub_text_list:
            # assert len(text) <= 510
            text_all_count += 1
            new_row = {}
            new_row['id'] = row['doc_id']
            new_row['text'] = text
            if len(text)>510: text_out_count+=1
            event_list = []
            if mode=="train": 
                for event in row["events"]:
                    new_event = {}
                    new_event["event_type"] = event["event_type"]
                    arguments = []

                    role_list = list(event.keys())
                    role_list.remove('event_type')
                    role_list.remove('event_id')
                    for role in role_list:
                        arg_all_count += 1
                        argument = event[role]
                        if argument == "":
                            continue
                        # print(argument)
                        argument_start_index_list = find_all(text, argument)
                        # print(argument_start_index_list)
                        if not argument_start_index_list:
                            # print(row, role)
                            continue
                        # if argument_start_index_list[0]> 510: 
                            # arg_out_count += 1
                        for argument_start_index in argument_start_index_list:
                            argument_map = {}
                            argument_map['role'] = role
                            argument_map['argument'] = argument
                            argument_map['argument_start_index']= argument_start_index
                            arguments.append(argument_map)
                    new_event['arguments'] = arguments
                    event_list.append(new_event)

            new_row['event_list'] = event_list
            res.append(new_row)
    # print(res[:10])
    print(text_all_count, text_out_count, arg_all_count, arg_out_count)
    write_file(res, output_file)

# 统计 event_type 分布
def data_analysis(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    label_list= get_labels(task='trigger', mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    label_count = [0 for i in range(len(label_list))]
    count = 0
    for row in rows:
        row = json.loads(row)
        text = row['text']
        if len(text) > 510:
            count += 1
        for event in row["event_list"]:
            event_type = event["event_type"]
            label_count[label_map[event_type]] += 1
    print(label_count)
    print(count)

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

def schema_analysis(path="./data/event_schema/event_schema.json"):
    rows = open(path, encoding='utf-8').read().splitlines()
    argument_map = {}
    for row in rows:
        d_json = json.loads(row)
        event_type = d_json["event_type"]
        for r in d_json["role_list"]:
            role = r["role"]
            if role in argument_map:
                argument_map[role].append(event_type)
            else: 
                argument_map[role]= [event_type]
    argument_unique = []
    argument_duplicate = []
    for argument, event_type_list in argument_map.items():
        if len(event_type_list)==1:
            argument_unique.append(argument)
        else:
            argument_duplicate.append(argument)

    print(argument_unique, argument_duplicate)
    for argument in argument_duplicate:
        print(argument_map[argument])

    return argument_map


if __name__ == '__main__':
    # labels = get_labels(path="./data/event_schema/event_schema.json", task='trigger', mode="classification")
    # print(len(labels), labels[50:60])
    
    convert("./data/ccks4_2/train_init.json", "./data/ccks4_2/train_split.json")
    # convert("./data/ccks4_2/dev_init.json", "./data/ccks4_2/dev_split.json", mode="dev")
    # data_val("./data/ccks4_2/train.json")

    # data_analysis("./data/ccks4_2/train.json")

    # 无异常
    # position_val("./data/train_data/train.json")

    # get_num_of_arguments("./results/test_pred_bin_segment.json")

    # read_write("./output/eval_pred.json", "./results/eval_pred.json")
    # read_write("./results/test1.trigger.pred.json", "./results/paddle.trigger.json")

    # schema_analysis()


