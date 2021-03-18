import json
import os
from utils import get_labels, write_file
from numpy import mean

def trigger_classify_file_remove_id(input_file, output_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        row.pop("id")
        row.pop("text")
        results.append(row)
    write_file(results,output_file)

def trigger_classify_process(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    count = 0
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        count += 1
        if "id" not in row:
            row["id"]=count
        labels = []
        if is_predict: 
            results.append({"id":row["id"], "text":row["text"], "labels":labels})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            labels.append(event_type)
        labels = list(set(labels))
        results.append({"id":row["id"], "text":row["text"], "labels":labels})
    write_file(results,output_file)


def role_process_segment_bio(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    len_text = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        len_text.append(len(row["text"]))
        if len(list(row["text"]))!= len(row["text"]):
            print("list and text mismatched")
        labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event["trigger_start_index"]
            segment_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                segment_ids[i] = 1

            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                labels[argument_start_index]= "B-{}".format(role)
                for i in range(1, len(argument)):
                    labels[argument_start_index+i]= "I-{}".format(role)
                # if arg['alias']!=[]: print(arg['alias'])
            
            results.append({"id":row["id"],  "event_type":event_type, "segment_ids":segment_ids,\
                 "tokens":list(row["text"]), "labels":labels})
    write_file(results,output_file)
    print(min(len_text), max(len_text), mean(len_text))


def role_process_segment_bin(input_file, output_file, is_predict=False):
    label_list = get_labels(task= "role", mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), \
                "start_labels":['O']*len(row["text"]), "end_labels":['O']*len(row["text"])})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event["trigger_start_index"]
            segment_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                segment_ids[i] = 1
            start_labels = ['O']*len(row["text"]) 
            end_labels = ['O']*len(row["text"]) 

            for arg in event["arguments"]:
                role = arg['role']
                role_id = label_map[role]
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1

                if start_labels[argument_start_index]=="O":
                    start_labels[argument_start_index] = role
                else: 
                    start_labels[argument_start_index] += (" "+ role)
                if end_labels[argument_end_index]=="O":
                    end_labels[argument_end_index] = role
                else: 
                    end_labels[argument_end_index] += (" "+ role)

                # if arg['alias']!=[]: print(arg['alias'])
            results.append({"id":row["id"], "tokens":list(row["text"]), "event_type":event_type, \
                "segment_ids":segment_ids,"start_labels":start_labels, "end_labels":end_labels})
    write_file(results,output_file)



def joint_process_bin(input_file, output_file, is_predict=False):
    label_list = get_labels(task= "role", mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        
        if is_predict:
            results.append({"id":row["id"], "tokens":list(row["text"]), \
              "trigger_start_labels":['O']*len(row["text"]), "role_end_labels":['O']*len(row["text"]), \
              "role_start_labels":['O']*len(row["text"]), "role_end_labels":['O']*len(row["text"])})
            continue

        trigger_start_labels = ['O']*len(row["text"]) 
        trigger_end_labels = ['O']*len(row["text"]) 

        # trigger
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event['trigger_start_index']
            trigger_end_index = trigger_start_index + len(trigger) -1
            if trigger_start_labels[trigger_start_index]=="O":
                trigger_start_labels[trigger_start_index] = event_type
            else: 
                trigger_start_labels[trigger_start_index] += (" "+ event_type)
            if trigger_end_labels[trigger_end_index]=="O":
                trigger_end_labels[trigger_end_index] = event_type
            else: 
                trigger_end_labels[trigger_end_index] += (" "+ event_type)
        
        # role
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event['trigger_start_index']
            segment_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                segment_ids[i] = 1

            role_start_labels = ['O']*len(row["text"]) 
            role_end_labels = ['O']*len(row["text"]) 

            for arg in event["arguments"]:
                role = arg['role']
                role_id = label_map[role]
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1
                
                if role_start_labels[argument_start_index]=="O":
                    role_start_labels[argument_start_index] = role
                else: 
                    role_start_labels[argument_start_index] += (" "+ role)
                    
                if role_end_labels[argument_end_index]=="O":
                    role_end_labels[argument_end_index] = role
                else: 
                    role_end_labels[argument_end_index] += (" "+ role)

                # if arg['alias']!=[]: print(arg['alias'])
            results.append({"id":row["id"], "tokens":list(row["text"]), "segment_ids":segment_ids, \
                "trigger_start_labels":trigger_start_labels, "trigger_end_labels":trigger_end_labels, \
                    "role_start_labels":role_start_labels, "role_end_labels":role_end_labels})

    write_file(results,output_file)



def role_process_filter(event_class, input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["text"])
        if is_predict: continue
        flag = False
        for event in row["event_list"]:
            event_type = event["event_type"]
            if event_class != event["class"]:
                continue
            flag = True
            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                labels[argument_start_index]= "B-{}".format(role)
                for i in range(1, len(argument)):
                    labels[argument_start_index+i]= "I-{}".format(role)
        if not flag: continue
        results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
    write_file(results,output_file)

def get_event_class(schema_file):
    rows = open(schema_file, encoding='utf-8').read().splitlines()
    labels=[]
    for row in rows:
        row = json.loads(row)
        event_class = row["class"]
        if event_class in labels:
            continue
        labels.append(event_class)
    return labels


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

    # trigger_classify_file_remove_id("./data/trigger_classify/dev.json", "./data/trigger_classify/dev_without_id.json")

 
    # trigger_classify_process("./data/ccks4_2/train.json", "./data/trigger_classify/train.json")
    # trigger_classify_process("./data/ccks4_2/dev.json", "./data/trigger_classify/test.json")

    # role_process_segment_bio("./data/FewFC-main/converted/train_trans.json", "./data/FewFC-main/role_trans_segment/train.json")
    # role_process_segment_bio("./data/FewFC-main/converted/test_trans.json", "./data/FewFC-main/role_trans_segment/test.json")

    # role_process_segment_bin("./data/train_data/train.json", "./data/role_segment_bin/train.json")
    # role_process_segment_bin("./data/dev_data/dev.json","./data/role_segment_bin/dev.json")

    # joint_process_bin("./data/train_data/train.json", "./data/joint_bin/train.json")
    # joint_process_bin("./data/dev_data/dev.json","./data/joint_bin/dev.json")
    # joint_process_bin("./data/test1_data/test1.json", "./data/joint_bin/test.json",is_predict=True)

    # split_data("./data/FewFC-main/trigger_base/train.json",  "./data/FewFC-main/trigger_base",  num_split=5)
    # split_data("./data/FewFC-main/trigger_trans/train.json",  "./data/FewFC-main/trigger_trans",  num_split=5)
    # split_data("./data/FewFC-main/role_base/train.json",  "./data/FewFC-main/role_base",  num_split=5)
    # split_data("./data/FewFC-main/role_trans/train.json",  "./data/FewFC-main/role_trans",  num_split=5)
    # split_data("./data/FewFC-main/role_trans_segment/train.json",  "./data/FewFC-main/role_trans_segment",  num_split=5)
    split_data("./data/FewFC-main/all/train.json",  "./data/FewFC-main/all",  num_split=5)

    # event_class_list = get_event_class("./data/event_schema/event_schema.json")
    # for event_class in event_class_list:
    #     if not os.path.exists("./data/role/{}".format(event_class)):
    #         os.makedirs("./data/role/{}".format(event_class))
    #     role_process_filter(event_class, "./data/train_data/train.json", "./data/role/{}/train.json".format(event_class))
    #     role_process_filter(event_class, "./data/dev_data/dev.json","./data/role/{}/dev.json".format(event_class))
