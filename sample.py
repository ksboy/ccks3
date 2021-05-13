import numpy as np
from numpy import mean
import random
np.random.seed(0)
random.seed(0)


def get_class_name(rawtag):
    # get (finegrained) class name
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return rawtag[2:]
    else:
        return rawtag

class Sample:
    def __init__(self, example):
        self.words, self.tags = example.words, example.labels
        # strip B-, I-
        self.normalized_tags = list(map(get_class_name, self.tags))
        self.entity_count = {}

    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.entity_count:
                        self.entity_count[current_tag] += 1
                    else:
                        self.entity_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.entity_count:
                self.entity_count[current_tag] += 1
            else:
                self.entity_count[current_tag] = 1

    def get_entity_count(self):
        if self.entity_count:
            return self.entity_count
        else:
            self.__count_entities__()
            return self.entity_count

    def get_tag_class(self):
        # strip 'B' 'I' 
        tag_class = list(set(self.normalized_tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def valid(self, target_classes):
        return True
        # return set(self.get_entity_count().keys()).intersection(set(target_classes)) and not set(self.get_entity_count().keys()).difference(set(target_classes))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])



class BatchSampler:
    def __init__(self, N, K, Q, samples, classes):
        self.K = K
        self.N = N
        self.Q = Q
        self.samples = samples
        self.classes= classes

    def __additem__(self, index, set_class):
        entity_count = self.samples[index].get_entity_count()
        for class_name in entity_count:
            if class_name in set_class:
                set_class[class_name] += entity_count[class_name]
            else:
                set_class[class_name] = entity_count[class_name]

    def __popitem__(self, index, set_class):
        entity_count = self.samples[index].get_entity_count()
        for class_name in entity_count:
            if class_name in set_class:
                set_class[class_name] -= entity_count[class_name]
            else:
                assert(0)

    def __valid_add_sample__(self, sample, set_class, target_classes):
        threshold = 3 * set_class['k']
        entity_count = sample.get_entity_count()
        if not entity_count:
            return False
        isvalid = False
        for class_name in entity_count:
            if class_name not in target_classes:
                return False
            elif class_name not in set_class:
                isvalid = True
            elif set_class[class_name] + entity_count[class_name] > threshold:
                return False
            elif set_class[class_name] < set_class['k']:
                isvalid = True
        return isvalid
    
    def __valid_pop_sample__(self, sample, set_class, target_classes):
        threshold = 10000 * set_class['k']
        entity_count = sample.get_entity_count()
        if not entity_count:
            return False
        isvalid = False
        for class_name in entity_count:
            if class_name not in target_classes:
                return False
            elif class_name not in set_class:
                isvalid = True
            elif set_class[class_name] > threshold:
                return False
            elif set_class[class_name] - entity_count[class_name] < set_class['k']:
                isvalid = True
        return isvalid

    def __finish__(self, set_class):
        if len(set_class) < self.N+1:
            return False
        for k in set_class:
            if set_class[k] < set_class['k']:
                return False
        return True 

    def __get_candidates__(self, target_classes):
        return [idx for idx, sample in enumerate(self.samples) if sample.valid(target_classes)]

    def sample(self):
        target_classes = random.sample(self.classes, self.N)
        support_class = {'k':self.K}
        # ['收购-number'] ['收购-sub-per'] ['签署合同-obj-per'] ['收购-proportion']
        support_idx = [180-1, 228-1] + [117-1, 220-1] + \
            [516-1, 531-1, 748-1, 768-1] + [37-1, 57-1]
        for index in support_idx:
            self.__additem__(index, support_class)

        query_class = {'k':self.Q}
        query_idx = []
        candidates = self.__get_candidates__(target_classes)
        # greedy search for support set
        step = 0
        while not self.__finish__(support_class) and step < 100000:
            step += 1
            index = random.sample(candidates, 1)[0]
            if index not in support_idx:
                if self.__valid_add_sample__(self.samples[index], support_class, target_classes):
                    self.__additem__(index, support_class)
                    support_idx.append(index)
        
        for index in support_idx:
            if not self.__valid_pop_sample__(self.samples[index], support_class, target_classes):
                self.__popitem__(index, support_class)
                support_idx.remove(index)

        return target_classes, support_idx

if __name__ == '__main__':

    from utils_ner_bio import read_examples_from_file
    from utils import get_labels
    data_dir = "./data/FewFC-main/rearranged/trans/"
    examples = read_examples_from_file(data_dir, mode='train', task="role", dataset="ccks")
    samples = [Sample(example) for example in examples]
    classes = get_labels(path="./data/FewFC-main/event_schema/trans.json", task='role', mode="classification")
    sampler = BatchSampler(31, 5, 5, samples, classes)
    target_classes, support_idx = sampler.sample()
    print(target_classes, support_idx)

    # support_idx = [179, 227, 116, 219, 515, 530, 747, 767, 36, 56, 323, 625, 655, 565, 488, 453, 533, 561, 14, 408, 727, 640, 626, 505, 249, 720, 581, 244, 556, 93, 520, 111, 560, 553, 818, 617, 601, 394, 297, 188, 191, 91, 695, 39, 716, 537, 603, 224, 587, 59, 80, 319, 158, 8, 304]
    # lines = open(data_dir+"train.json", encoding='utf-8').read().splitlines()
    # res =[lines[index]+"\n" for index in support_idx]
    # outf = open(data_dir+"support.json", "w")
    # outf.writelines(res)


