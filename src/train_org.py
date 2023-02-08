from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import logging
import os
from src.custom_loader import custom_loader
import argparse
import random
from sklearn.metrics import precision_score,recall_score,f1_score

parser = argparse.ArgumentParser(description='ハイパラ')
parser.add_argument('--epoch', help = 'epoch',type=int,default=100)
parser.add_argument('--num_classes',type=int,default=3)
parser.add_argument('--data_path', metavar='ARCH', default='./clst20_maha_custom_028data',help='model architecture:')
parser.add_argument('--aug',type=int,default=0)
parser.add_argument('--seed',type=int,default=9999)
args = parser.parse_args()

"""シード値の固定"""
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    
"""###実行結果などを残すフォルダの作成"""
#直下にフォルダを作成
path = args.data_path
result_path = f"./{path[2:9]}_{path[10:13]}_org_result_aug:{args.aug}_seed:{args.seed}"
log_path = result_path+'/log'
log_img_path = result_path+'/log_img'
os.makedirs(result_path,exist_ok=True)
os.makedirs(log_path,exist_ok=True)
os.makedirs(log_img_path,exist_ok=True)

#logのtxtファイル作成
f = open(log_path+'/log.txt','w')
f.close()
# logを残すための関数。'w'は上書きオプション
logging.basicConfig(
    filename=log_path+'/log.txt',
    filemode='w',
    format="%(asctime)s %(message)s",
    level =logging.INFO)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

loader = custom_loader(args.data_path,16,args)
train_loader = loader.run('train')
valid_loader = loader.run('valid')
test_loader = loader.run('test')
org_train_loader = loader.run('org_train')


model_org=models.resnet18(pretrained=False,num_classes=args.num_classes).to(device)
gpu_num = torch.cuda.device_count()
if(torch.cuda.device_count()>1): 
    model_org = torch.nn.DataParallel(model_org,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())

opt_org=optim.SGD(model_org.parameters(),lr=0.1,weight_decay=0.001)

# org_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_org,patience=5,factor=0.1)
org_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_org,T_max=10,eta_min=0.0001)

criterion = nn.CrossEntropyLoss()

history={
    'org_train_loss':[],
    'org_train_acc':[],
    'org_valid_loss':[],
    'org_valid_acc':[],
}

epochs = args.epoch
best_loss=0
for epoch in range(epochs):

    """オリジナルデータの方の学習"""
    loop_org = tqdm(org_train_loader, unit='batch', desc='| Org Train | Epoch {:>3} |'.format(epoch+1))
    model_org.train()
    scaler_org = torch.cuda.amp.GradScaler()
    batch_loss = []
    batch_acc  = []
    for batch in loop_org:  
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # import pdb;pdb.set_trace()

        opt_org.zero_grad()       
        with torch.cuda.amp.autocast():
            out = model_org(x)             
            loss = criterion(out, label)    
        scaler_org.scale(loss).backward()
        scaler_org.step(opt_org)
        scaler_org.update()
       
        pred_class = torch.argmax(out, dim=1)       
        acc = torch.sum(pred_class == label)/len(label) 

        batch_loss.append(loss)
        batch_acc.append(acc)

    org_train_avg_loss = torch.tensor(batch_loss).mean()
    org_train_avg_acc  = torch.tensor(batch_acc).mean()

    #========== 検証用データへの処理 ==========#
    # モデルを評価モードに設定
    model_org.eval()

    custom_batch_loss = []
    custom_batch_acc = []
    org_batch_loss = []
    org_batch_acc = []
    rand_batch_loss = []
    rand_batch_acc = []

    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in valid_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out_org = model_org(x)

            loss_org = criterion(out_org, label)
            org_pred_class = torch.argmax(out_org, dim=1)
            org_acc = torch.sum(org_pred_class == label)/len(label)

            org_batch_loss.append(loss_org)
            org_batch_acc.append(org_acc)
        org_valid_avg_loss = torch.tensor(org_batch_loss).mean()
        org_valid_avg_acc  = torch.tensor(org_batch_acc).mean()
    
        if epoch == 1: 
            torch.save(model_org.state_dict(), result_path + '/ex5_weights.pth')
            best_loss = org_valid_avg_loss
        if best_loss > org_valid_avg_loss:
            best_loss = org_valid_avg_loss
            torch.save(model_org.state_dict(), result_path + '/ex5_weights.pth')

    history['org_train_loss'].append(org_train_avg_loss)
    history['org_valid_loss'].append(org_valid_avg_loss)
    history['org_train_acc'].append(org_train_avg_acc)
    history['org_valid_acc'].append(org_valid_avg_acc)
    
    logging.info(f"| Org_Train | Epoch   {epoch+1} |: org_train_loss:{org_train_avg_loss:.3f}, org_train_acc:{org_train_avg_acc*100:3.3f}% | org_valid_loss:{org_valid_avg_loss:.5f}, org_valid_acc:{org_valid_avg_acc*100:3.3f}%")
    print(f"| Org_Train | Epoch   {epoch+1} |: org_train_loss:{org_train_avg_loss:.3f}, org_train_acc:{org_train_avg_acc*100:3.3f}% | org_valid_loss:{org_valid_avg_loss:.5f}, org_valid_acc:{org_valid_avg_acc*100:3.3f}%")

org_filename_acc = log_img_path+ "/org_Accuracy.png"
org_filename_loss = log_img_path+ "/org_loss.png"
fig3 = plt.figure()
plt.plot(history['org_train_acc'])
plt.plot(history['org_valid_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['org train acc', 'org validation acc'], loc='lower right')
plt.grid(True)
fig3.savefig(org_filename_acc)
# Lossを描画
fig4 = plt.figure()
plt.plot(history['org_train_loss'])
plt.plot(history['org_valid_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['org train loss', 'org validation loss'], loc='upper right')
plt.grid(True)
fig4.savefig(org_filename_loss)


org_batch_loss = []
org_batch_acc = []
pred = np.array([])
true = np.array([])


model_org = models.resnet18(pretrained=False,num_classes=args.num_classes)                                    # 保存時と全く同じ構成のモデルを構築
params = torch.load(result_path + '/ex5_weights.pth', map_location=device)   # 保存した重みを読み出す
model_org.load_state_dict(params) 
model_org.to(device)
model_org.eval()


# 勾配計算に必要な情報を記録しないよう設定
with torch.no_grad():
    loop = tqdm(test_loader, unit='batch', desc='| Test | Epoch {:>3} |'.format(epochs+1))
    for batch in loop:
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        out_org = model_org(x)
        org_pred_class = torch.argmax(out_org, dim=1)
        acc_org = torch.sum(org_pred_class == label)/len(label)
        org_batch_acc.append(acc_org)

        pred = np.append(pred,org_pred_class.to('cpu').detach().numpy().copy())
        true = np.append(true,label.to('cpu').detach().numpy().copy())

    org_test_avg_loss = torch.tensor(org_batch_loss).mean()
    org_test_avg_acc  = torch.tensor(org_batch_acc).mean()
    
print(f'precision:{precision_score(true,pred,zero_division=True)}')
print(f'recall:{recall_score(true,pred)}')
print(f'f1:{f1_score(true,pred)}')
print(f"org_test_loss   ：{org_test_avg_loss:3.5f}")
print(f"org_test_acc    ：{org_test_avg_acc*100:2.3f}%")
logging.info(f"org_test_loss   ：{org_test_avg_loss:3.5f}")
logging.info(f"org_test_acc    ：{org_test_avg_acc*100:2.3f}%")
logging.info(f'precision:{precision_score(true,pred,zero_division=True)}')
logging.info(f'recall:{recall_score(true,pred)}')
logging.info(f'f1:{f1_score(true,pred)}')


