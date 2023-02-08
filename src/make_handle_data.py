import numpy as np
import random
import torch
import torchvision.transforms.functional as functional
import torchvision.models as models
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import os
import handle_dataloader as handle
from coreset import furthest_first



"""シード値の固定"""
random_seed = 9999
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

loader=handle.handle_loader('/home/oshita/cleansing/data/handle/gray',16)
train_loader = loader.run("train")
valid_loader = loader.run("valid")
test_loader = loader.run("test")


model = models.resnet18(pretrained=True)
model.to(device)
model.eval()
gpu_num = torch.cuda.device_count()
if(gpu_num>1): 
    model = torch.nn.DataParallel(model,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())

feat = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device)
}
train_data = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device)
}

trainloop = tqdm(train_loader, unit='batch', desc='| train | Epoch {:>3} |'.format(1))

with torch.no_grad():
    for data,labels in trainloop:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feature=model(data)
        for cls in range(2):
            train_data[cls] = torch.cat((train_data[cls],data[labels==cls]))
            feat[cls] = torch.cat((feat[cls],feature[labels==cls]))

valid_data = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device)
}

validloop = tqdm(valid_loader, unit='batch', desc='| valid | Epoch {:>3} |'.format(1))
with torch.no_grad():
    for data,labels in validloop:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(2):
            valid_data[cls] = torch.cat((valid_data[cls],data[labels==cls]))

test_data = {
        0:torch.tensor([]).to(device), 1:torch.tensor([]).to(device)
}


testloop = tqdm(test_loader, unit='batch', desc='| test | Epoch {:>3} |'.format(1))
with torch.no_grad():
    for data,labels in testloop:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for cls in range(2):
            test_data[cls] = torch.cat((test_data[cls],data[labels==cls]))


target=['ok','ng']
_class = ['ok','ng']
cluster=[20,20]

org_tr_data=torch.tensor([]).to(device)
org_tr_labels=torch.tensor([]).to(device)
tr_data=torch.tensor([]).to(device)
va_data=torch.tensor([]).to(device)
te_data=torch.tensor([]).to(device)
tr_labels=torch.tensor([]).to(device)
va_labels=torch.tensor([]).to(device)
te_labels=torch.tensor([]).to(device)


del_num=500
for name in target:
    index=_class.index(name)
    num_clusters = cluster[index]
    _feat=feat[index]
    # model = GaussianMixture(n_components=num_clusters,max_iter=30,random_state=random_seed)
    # model.fit(_feat)
    # pred = model.predict(_feat)
    # for clst in range(num_clusters):
        # if(train_data[index][pred==clst].shape[0]>del_num):
        #     print('----cut sample num----')
        #     print(f'size:{train_data[index][pred==clst].shape[0]}')
        #     data_num = train_data[index][pred==clst].shape[0]
        #     tr_data=torch.cat((tr_data,train_data[index][pred==clst][0:del_num]),dim=0)
        #     tr_labels=torch.cat((tr_labels,torch.tensor([target.index(name)]).repeat(del_num).to(device)),dim=0)
        # if(train_data[index][pred==clst].shape[0]>del_num):
        #     print('----cut sample num----')
        #     print(f'size:{train_data[index][pred==clst].shape[0]}')
        #     _feat=_feat[pred==clst]
        #     _model = GaussianMixture(n_components=20,max_iter=30,random_state=random_seed)
        #     _model.fit(_feat)
        #     _pred = _model.predict(_feat)
        #     for i in np.unique(_pred):
        #         if(train_data[index][pred==clst][_pred==i].shape[0] > 50):
        #             tr_data=torch.cat((tr_data,train_data[index][pred==clst][_pred==i]),dim=0)
        #             tr_labels=torch.cat((tr_labels,torch.tensor([target.index(name)]).repeat(train_data[index][pred==clst][_pred==i].shape[0]).to(device)),dim=0)
        # else:
        #     tr_data=torch.cat((tr_data,train_data[index][pred==clst]),dim=0)
        #     tr_labels=torch.cat((tr_labels,torch.tensor(target.index(name)).repeat(train_data[index][pred==clst].shape[0]).to(device)),dim=0)
    if name == 'ok':
        print(torch.var(_feat).item())
        badget = int(len(_feat)*0.6)
        idx = furthest_first(_feat,False,badget,20)
        print(torch.var(_feat[idx]).item())
        tr_data = torch.cat((tr_data,train_data[index][idx]),dim=0)
        tr_labels = torch.cat((tr_labels,torch.tensor(target.index(name)).repeat(train_data[index][idx].shape[0]).to(device)))
        # import pdb;pdb.set_trace()
    else:
        tr_data = torch.cat((tr_data,train_data[index]),dim=0)
        tr_labels = torch.cat((tr_labels,torch.tensor(target.index(name)).repeat(train_data[index].shape[0]).to(device)))
    org_tr_data = torch.cat((org_tr_data,train_data[index]),dim=0)
    org_tr_labels=torch.cat((org_tr_labels,torch.tensor(target.index(name)).repeat(train_data[index].shape[0]).to(device)),dim=0)
    va_data = torch.cat((va_data,valid_data[index]),dim=0)
    va_labels=torch.cat((va_labels,torch.tensor([target.index(name)]).repeat(valid_data[index].shape[0]).to(device)),dim=0)
    te_data = torch.cat((te_data,test_data[index]),dim=0)
    te_labels=torch.cat((te_labels,torch.tensor([target.index(name)]).repeat(test_data[index].shape[0]).to(device)),dim=0)
        
print(org_tr_data.shape)
print(tr_data.shape)
print(va_data.shape)
print(te_data.shape)

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.20, 0.20, 0.20]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

for dir_name in ['train','valid','test','org_train','random_select_train']:
    for i in range(2):
        os.makedirs(f'./coreset_0.6_data/{dir_name}/{i}',exist_ok=True)
count=0
for img,label in zip(tr_data,tr_labels):
    label = int(label.item())
    tr_img=functional.to_pil_image(unnorm(img))
    tr_img.save(f'./coreset_0.6_data/train/{label}/{count}.png')
    count+=1
count=0
for img,label in zip(va_data,va_labels):
    label = int(label.item())
    va_img=functional.to_pil_image(unnorm(img))
    va_img.save(f'./coreset_0.6_data/valid/{label}/{count}.png')
    count+=1
count=0
for img,label in zip(te_data,te_labels):
    label = int(label.item())
    te_img=functional.to_pil_image(unnorm(img))
    te_img.save(f'./coreset_0.6_data/test/{label}/{count}.png')
    count+=1
count=0
for img,label in zip(org_tr_data,org_tr_labels):
    label = int(label.item())
    org_tr_img=functional.to_pil_image(unnorm(img))
    org_tr_img.save(f'./coreset_0.6_data/org_train/{label}/{count}.png')
    count+=1

"""randomにサンプリング．枚数はtrainに合わせる"""
import glob
import shutil
path0 = glob.glob('./coreset_0.6_data/org_train/0/*')
path1 = glob.glob('./coreset_0.6_data/org_train/1/*')
path_tr0 = glob.glob('./coreset_0.6_data/train/0/*')
path_tr1 = glob.glob('./coreset_0.6_data/train/1/*')
random_path0 = random.sample(path0,len(path_tr0))
random_path1 = random.sample(path1,len(path_tr1))
for p in random_path0:
    shutil.copy(p,'./coreset_0.6_data/random_select_train/0')
for p in random_path1:
    shutil.copy(p,'./coreset_0.6_data/random_select_train/1')




