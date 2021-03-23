import json
import collections
from utils import get_labels

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

# 统计 event_type 分布
def data_analysis(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    label_list= get_labels(task='trigger', mode="classification")
    print(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}
    label_count = [0 for i in range(len(label_list))]
    count = 0
    len_list = []
    for row in rows:
        row = json.loads(row)
        text = row['text']
        if len(text) > 254:
            count += 1
        len_list.append(len(text))
        for event in row["event_list"]:
            event_type = event["event_type"]
            label_count[label_map[event_type]] += 1

    obj = collections.Counter(len_list)
    # print([[k,v] for k,v in obj.items()])
    print(sorted([[k,v] for k,v in obj.items()], key=lambda x:x[0])[-10:])
    print(label_count)
    print(count)


def data_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()

    event_type_count = 0
    event_type_count2 = 0
    role_count = 0
    arg_count = 0
    arg_count2 = 0
    arg_role_count = 0
    arg_role_one_event_count = 0
    trigger_count = 0 # triggr

    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)

        arg_start_index_list=[]
        arg_start_index_map={}
        event_type_list = []
        trigger_start_index_list = []

        event_type_flag = False  # 两个不同类型的事件
        event_type_flag2 = False # 两个事件
        arg_start_index_flag = False # 一个事件，一个 role 对应多个 arg: 相同/不同
        arg_start_index_flag2 = False # 一个事件，一个 role 对应多个 arg：不同
        role_flag = False # 一个/多个事件，一个arg对应多个role：相同/不同
        arg_role_flag= False # 一个/多个事件，一个arg对应多个role：不同
        arg_role_one_event_flag= False # 一个事件 一个arg对应多个role：不同
        trigger_flag = False

        for event in row["event_list"]:
            event_type = event["event_type"]
            if event_type_list==[]: 
                event_type_list.append(event_type)
            else:
                event_type_flag2 = True
                # print(row)
                if event_type not in event_type_list:
                    # event_class_count += 1
                    event_type_flag = True
                    # print(row)
            
            trigger = event["trigger"]
            trigger_start_index= event["trigger_start_index"]
            if [trigger_start_index, trigger] not in trigger_start_index_list:
                trigger_start_index_list.append([trigger_start_index, trigger])
            else:
                trigger_flag = True
                # print(row)

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
                    else:
                        pass
                        # print(row)
                
                if argument_start_index not in arg_start_index_map_in_one_event:
                    arg_start_index_map_in_one_event[argument_start_index]= role
                else:
                    if role!= arg_start_index_map_in_one_event[argument_start_index]:
                        arg_role_one_event_flag = True
                        # if event_type != "股份股权转让":
                            # print(row)
                        # return 0


                if argument_start_index not in arg_start_index_list:
                    arg_start_index_list.append(argument_start_index)
                    arg_start_index_map[argument_start_index]= role
                else: 
                    role_flag = True
                    if role!= arg_start_index_map[argument_start_index]:
                        arg_role_flag = True
                        if len(row["event_list"])>1: print(row)
                    else:
                        pass
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
            # print(row)
        if trigger_flag:
            trigger_count += 1
    
    print(event_type_count, event_type_count2, role_count, arg_count, arg_count2, arg_role_count, arg_role_one_event_count, trigger_count)

if __name__ == '__main__':

    data_val("./data/FewFC-main/converted/train_base.json")
    # data_analysis("./data/trans/train.json")
    # data_analysis("./data/trans/test.json")

    # schema_analysis()