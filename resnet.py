import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class resnet(nn.Module):
    def __init__(self,layer=2,coreset='coreset'):
        super().__init__()
        self.layer = layer
        self.coreset=coreset
        self.model = models.resnet18(pretrained=True)
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool


    def forward(self, x):
        if self.layer==1: 
            out = self.layer1(self.relu(self.bn1(self.conv1(x))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        elif self.layer==2: 
            out = self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        elif self.layer==3: 
            out = self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x))))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        else:
            raise ValueError('Invalid layer is selected')

class ssl_resnet(nn.Module):
    def __init__(self,layer=2,coreset='coreset'):
        super().__init__()
        self.layer = layer
        self.coreset=coreset
        self.model = model = models.resnet18(pretrained=False)
        device = torch.device('cuda')
        params = torch.load('/home/oshita/cleansing/my_project/simsiam/checkpoint/checkpoint_ver.pth.tar', map_location=device)  
        model.load_state_dict(params['state_dict'],strict=False) 
        
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool

    def forward(self, x):
        if self.layer==1: 
            out = self.layer1(self.relu(self.bn1(self.conv1(x))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        elif self.layer==2: 
            out = self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        elif self.layer==3: 
            out = self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x))))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        elif self.layer==4: 
            out = self.model.layer4(self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))))
            if self.coreset=='coreset': 
                out = self.avgpool(out)
                return out
            elif self.coreset=='greedy_coreset': 
                return out
        else:
            raise ValueError('Invalid layer is selected')