from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("/home/mhxia/whou/workspace/pretrained_models/roberta-base/")
    # "/home/mhxia/whou/workspace/pretrained_models/roberta-base/vocab.json", "/home/mhxia/whou/workspace/pretrained_models/roberta-base/merges.txt")
res= tokenizer.tokenize("William?")
print(res)