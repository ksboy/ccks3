import json
import os
from utils import get_labels, write_file
from numpy import mean

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
        # assert len(text) <= 510
        text_all_count += 1
        new_row = {}
        new_row['id'] = row['id']
        text = row['content']
        new_row['text'] = text
        if len(text)>510: text_out_count+=1
        event_list = []
        if mode=="train": 
            for event in row["events"]:
                new_event = {}
                new_event["event_type"] = event["type"]
                arguments = []

                for mention in event["mentions"]:
                    arg_all_count += 1
                    argument = mention["word"]
                    role = mention["role"]
                    argument_start_index = mention["span"][0]
                    if role == "trigger":
                        new_event["trigger"] = argument
                        new_event["trigger_start_index"] = argument_start_index
                        continue

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


def split_data(input_file, output_dir, num_split=5):
    datas = open(input_file, encoding='utf-8').read().splitlines()
    for i in range(num_split):
        globals()["train_data"+str(i+1)] = []
        globals()["dev_data"+str(i+1)] = []
    for i, data in enumerate(datas):
        cur = i % num_split + 1
        for j in range(num_split):
            if cur == j+1:
                globals()["dev_data" + str(j + 1)].append(json.loads(data))
            else:
                globals()["train_data"+str(j + 1)].append(json.loads(data))
    for i in range(num_split):
        cur_dir = os.path.join(output_dir, str(i))
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        write_file(globals()["train_data"+str(i + 1)], os.path.join(cur_dir, "train.json"))
        write_file(globals()["dev_data"+str(i + 1)], os.path.join(cur_dir, "dev.json"))


if __name__ == '__main__':
    # convert("./data/FewFC-main/rearranged/train_base.json", "./data/FewFC-main/converted/train_base.json")
    # convert("./data/FewFC-main/rearranged/test_base.json", "./data/FewFC-main/converted/test_base.json")
    # convert("./data/FewFC-main/rearranged/train_trans.json", "./data/FewFC-main/converted/train_trans.json")
    # convert("./data/FewFC-main/rearranged/test_trans.json", "./data/FewFC-main/converted/test_trans.json")
    
    # split_data("./data/FewFC-main/trigger_base/train.json",  "./data/FewFC-main/trigger_base",  num_split=5)
    # split_data("./data/FewFC-main/trigger_trans/train.json",  "./data/FewFC-main/trigger_trans",  num_split=5)
    # split_data("./data/FewFC-main/role_base/train.json",  "./data/FewFC-main/role_base",  num_split=5)
    # split_data("./data/FewFC-main/role_trans/train.json",  "./data/FewFC-main/role_trans",  num_split=5)
    # split_data("./data/FewFC-main/role_trans_segment/train.json",  "./data/FewFC-main/role_trans_segment",  num_split=5)
    split_data("./data/DuEE_1_0/train.json",  "./data/DuEE_1_0/base",  num_split=5)

