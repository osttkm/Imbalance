import torch 
import torch.nn as nn
import torchvision.models as models
import random
import numpy as np

def initialize_weights(m):
    if isinstance(m, nn.Conv2d): # Convolution層が引数に渡された場合
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu') # kaimingの初期化
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)   # bias項は0に初期化
    elif isinstance(m, nn.BatchNorm2d):         # BatchNormalization層が引数に渡された場合
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):              # 全結合層が引数に渡された場合
        nn.init.kaiming_normal_(m.weight.data)  # kaimingの初期化
        nn.init.constant_(m.bias.data, 0)       # biasは0に初期化

def get_ssl_model(device):
    model = models.resnet18(pretrained=False)
    params = torch.load('/home/oshita/cleansing/my_project/simsiam/checkpoint/checkpoint_ver.pth.tar', map_location=device)  
    model.load_state_dict(params['state_dict'],strict=False)
    model.fc = nn.Linear(model.fc.in_features,2)
    model.to(device)
    return model

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.20, 0.20, 0.20]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img
    


