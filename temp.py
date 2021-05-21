import torch
model = torch.load("/home/banifeng/ccks3/models/pretrain_models/ai_keywords/ai_keywords_model/pytorch_model.bin", map_location="cpu")
print(model.keys())