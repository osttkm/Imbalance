import numpy as np
import random
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as functional
import torchvision.models as models
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from torchvision.datasets import CIFAR10
import pickle
import os
import matplotlib.pyplot as plt



"""シード値の固定"""
random_seed = 9999
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

def load_cifar10(transform, train:bool):
    cifar10_dataset = CIFAR10(
                        root='/home/oshita/cleansing/data/cifar-10',
                        train=train,
                        download=False,
                        transform=transform
                        )
    return cifar10_dataset
    
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2])
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2])
    ]
)

dataset = load_cifar10(transform=train_transform, train=True)
test_dataset = load_cifar10(transform=test_transform, train=False)
valid_ratio = 0.1                               # 検証用データの割合を指定
valid_size = int(len(dataset) * valid_ratio)    # 検証用データの数
train_size = len(dataset) - valid_size 
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
print(f'train_data:{len(train_dataset)}')
print(f'test_data:{len(test_dataset)}')
train_loader = DataLoader(
    train_dataset,          # データセットを指定
    batch_size=256,          # バッチサイズを指定
    shuffle=True,           # シャッフルの有無を指定
    drop_last=True,         # バッチサイズで割り切れないデータの使用の有無を指定
    pin_memory=True,        # 少しだけ高速化が期待できるおまじない
    num_workers=4           # DataLoaderのプロセス数を指定
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=256,
    shuffle=False,
    pin_memory=True,
    num_workers=4   
)
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    pin_memory=True,
    num_workers=4   
)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

model = models.vit_b_16(weights=torch.load('./models/vit_model.pth'))
model.to(device)
model.eval()
gpu_num = torch.cuda.device_count()
if(gpu_num>1): 
    model = torch.nn.DataParallel(model,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())

feat = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device),2:torch.tensor([]).to(device),
        3:torch.tensor([]).to(device),4:torch.tensor([]).to(device),5:torch.tensor([]).to(device),
        6:torch.tensor([]).to(device),7:torch.tensor([]).to(device),8:torch.tensor([]).to(device),
        9:torch.tensor([]).to(device)
}
train_data = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device),2:torch.tensor([]).to(device),
        3:torch.tensor([]).to(device),4:torch.tensor([]).to(device),5:torch.tensor([]).to(device),
        6:torch.tensor([]).to(device),7:torch.tensor([]).to(device),8:torch.tensor([]).to(device),
        9:torch.tensor([]).to(device)
}
trainloop = tqdm(train_loader, unit='batch', desc='| train | Epoch {:>3} |'.format(1))

with torch.no_grad():
    for data,labels in trainloop:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feature=model(data)
        for cls in range(10):
            if(cls==3 or cls==4 or cls==6):
                train_data[cls] = torch.cat((train_data[cls],data[labels==cls]))
                feat[cls] = torch.cat((feat[cls],feature[labels==cls]))

with open("./feature.pkl","wb") as f:
    pickle.dump(feat, f)

valid_data = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device),2:torch.tensor([]).to(device),
        3:torch.tensor([]).to(device),4:torch.tensor([]).to(device),5:torch.tensor([]).to(device),
        6:torch.tensor([]).to(device),7:torch.tensor([]).to(device),8:torch.tensor([]).to(device),
        9:torch.tensor([]).to(device)
}
validloop = tqdm(valid_loader, unit='batch', desc='| valid | Epoch {:>3} |'.format(1))
with torch.no_grad():
    for data,labels in validloop:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(10):
                if(cls==3 or cls==4 or cls==6):
                        valid_data[cls] = torch.cat((valid_data[cls],data[labels==cls]))

test_data = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device),2:torch.tensor([]).to(device),
        3:torch.tensor([]).to(device),4:torch.tensor([]).to(device),5:torch.tensor([]).to(device),
        6:torch.tensor([]).to(device),7:torch.tensor([]).to(device),8:torch.tensor([]).to(device),
        9:torch.tensor([]).to(device)
}
testloop = tqdm(test_loader, unit='batch', desc='| test | Epoch {:>3} |'.format(1))
with torch.no_grad():
    for data,labels in testloop:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(10):
                if(cls==3 or cls==4 or cls==6):
                        test_data[cls] = torch.cat((test_data[cls],data[labels==cls]))

with open('feature.pkl', 'rb') as f:
    feat = pickle.load(f)


target=['cat','deer','flog']
_class = ['plane','car','bird','cat','deer','dog','flog','horse','ship','truck']
# cluster=[30,30,30,30,30,30,30,30,30,30]
cluster=[20,20,20,20,20,20,20,20,20,20]

org_tr_data=torch.tensor([]).to(device)
org_tr_labels=torch.tensor([]).to(device)
tr_data=torch.tensor([]).to(device)
va_data=torch.tensor([]).to(device)
te_data=torch.tensor([]).to(device)
tr_labels=torch.tensor([]).to(device)
va_labels=torch.tensor([]).to(device)
te_labels=torch.tensor([]).to(device)
del_num=0
for name in target:
    index=_class.index(name)
    num_clusters = cluster[index]
    del_num = 300
    _feat=feat[index].to('cpu').detach().numpy().copy()
    model = GaussianMixture(n_components=num_clusters,max_iter=30,random_state=random_seed)
    model.fit(_feat)
    pred = model.predict(_feat)

    for clst in range(num_clusters):
        import pdb;pdb.set_trace()
        if(train_data[index][pred==clst].shape[0]>del_num):
            print('----cut sample num----')
            print(f'size:{train_data[index][pred==clst].shape[0]}')
            tr_data=torch.cat((tr_data,train_data[index][pred==clst][0:del_num]),dim=0)
            tr_labels=torch.cat((tr_labels,torch.tensor([target.index(name)]).repeat(del_num).to(device)),dim=0)
        else:
            tr_data=torch.cat((tr_data,train_data[index][pred==clst]),dim=0)
            tr_labels=torch.cat((tr_labels,torch.tensor(target.index(name)).repeat(train_data[index][pred==clst].shape[0]).to(device)),dim=0)
        org_tr_data = torch.cat((org_tr_data,train_data[index][pred==clst]),dim=0)
        org_tr_labels=torch.cat((org_tr_labels,torch.tensor(target.index(name)).repeat(train_data[index][pred==clst].shape[0]).to(device)),dim=0)
    va_data = torch.cat((va_data,valid_data[index]),dim=0)
    va_labels=torch.cat((va_labels,torch.tensor([target.index(name)]).repeat(valid_data[index].shape[0]).to(device)),dim=0)
    te_data = torch.cat((te_data,test_data[index]),dim=0)
    te_labels=torch.cat((te_labels,torch.tensor([target.index(name)]).repeat(test_data[index].shape[0]).to(device)),dim=0)
        
print(org_tr_data.shape)
print(tr_data.shape)
print(va_data.shape)
print(te_data.shape)
# import pdb;pdb.set_trace()

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.20, 0.20, 0.20]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

for dir_name in ['train','valid','test','org_train','random_select_train']:
    for i in range(3):
        os.makedirs(f'./clst20_maha_custom_346data/{dir_name}/{i}',exist_ok=True)
count=0
for img,label in zip(tr_data,tr_labels):
    label = int(label.item())
    tr_img=functional.to_pil_image(unnorm(img))
    tr_img.save(f'./clst20_maha_custom_346data/train/{label}/{count}.png')
    count+=1
count=0
for img,label in zip(va_data,va_labels):
    label = int(label.item())
    va_img=functional.to_pil_image(unnorm(img))
    va_img.save(f'./clst20_maha_custom_346data/valid/{label}/{count}.png')
    count+=1
count=0
for img,label in zip(te_data,te_labels):
    label = int(label.item())
    te_img=functional.to_pil_image(unnorm(img))
    te_img.save(f'./clst20_maha_custom_346data/test/{label}/{count}.png')
    count+=1
count=0
for img,label in zip(org_tr_data,org_tr_labels):
    label = int(label.item())
    org_tr_img=functional.to_pil_image(unnorm(img))
    org_tr_img.save(f'./clst20_maha_custom_346data/org_train/{label}/{count}.png')
    count+=1

"""以下はサンプリング数をチェックしてから別のファイルで実行"""
# import glob
# path0 = glob.glob('/home/oshita/cleansing/my_project/maha_custom_028data/org_train/0/*')
# path1 = glob.glob('/home/oshita/cleansing/my_project/maha_custom_028data/org_train/1/*')
# path2 = glob.glob('/home/oshita/cleansing/my_project/maha_custom_028data/org_train/2/*')
# random_path0 = random.sample(path0,4334)
# random_path1 = random.sample(path1,4296)
# random_path2 = random.sample(path2,4397)
# for p in random_path0:
#     shutil.copy(p,'/home/oshita/cleansing/my_project/maha_custom_028data/random_select_train/0')
# for p in random_path1:
#     shutil.copy(p,'/home/oshita/cleansing/my_project/maha_custom_028data/random_select_train/1')
# for p in random_path2:
#     shutil.copy(p,'/home/oshita/cleansing/my_project/maha_custom_028data/random_select_train/2')




