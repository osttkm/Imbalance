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
parser.add_argument('--num_classes',help='num classes',type=int,default=3)
parser.add_argument('--data_path', type=str, default='clst20_handle_ng0.5_del600_data',help='model architecture:')
parser.add_argument('--aug',type=int,default=0)
parser.add_argument('--seed',help='seed',type=int,default=9999)
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
result_path = f"./{path[2:9]}_{path[10:13]}_custom_result_aug:{args.aug}_seed:{args.seed}"
log_path = result_path+'/log'
log_img_path = result_path+'/log_img'
os.makedirs(result_path,exist_ok=True)
os.makedirs(log_path,exist_ok=True)
os.makedirs(log_img_path,exist_ok=True)
f = open(log_path+'/log.txt','w')
f.close()
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

model_custom=models.resnet18(pretrained=False,num_classes=args.num_classes).to(device)
gpu_num = torch.cuda.device_count()
if(gpu_num>1): 
    model_custom = torch.nn.DataParallel(model_custom,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())

opt_custom=optim.SGD(model_custom.parameters(),lr=0.1,weight_decay=0.001)

# custom_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_custom,patience=5,factor=0.1)
custom_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_custom,T_max=10,eta_min=0.0001)
criterion = nn.CrossEntropyLoss()

history={
    'custom_train_loss':[],
    'custom_valid_loss':[],
    'custom_train_acc':[],
    'custom_valid_acc':[]
}
best_loss=0
epochs = args.epoch
for epoch in range(epochs):
    loop_custom = tqdm(train_loader, unit='batch', desc='| Custom Train | Epoch {:>3} |'.format(epoch+1))
    model_custom.train()
    scaler_custom = torch.cuda.amp.GradScaler()
    batch_loss = []
    batch_acc  = []
    for batch in loop_custom:  
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # import pdb;pdb.set_trace()

        opt_custom.zero_grad()       
        with torch.cuda.amp.autocast():
            out = model_custom(x)             
            loss = criterion(out, label)    
        scaler_custom.scale(loss).backward()
        scaler_custom.step(opt_custom)
        scaler_custom.update()
       
        pred_class = torch.argmax(out, dim=1)       
        acc = torch.sum(pred_class == label)/len(label) 

        batch_loss.append(loss)
        batch_acc.append(acc)

    custom_train_avg_loss = torch.tensor(batch_loss).mean()
    custom_train_avg_acc  = torch.tensor(batch_acc).mean()



    #========== 検証用データへの処理 ==========#
    # モデルを評価モードに設定
    model_custom.eval()
    custom_batch_loss = []
    custom_batch_acc = []


    with torch.no_grad():
        for batch in valid_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out_custom = model_custom(x)
            loss_custom = criterion(out_custom, label)

            custom_pred_class = torch.argmax(out_custom, dim=1)
            custom_acc = torch.sum(custom_pred_class == label)/len(label)
            custom_batch_loss.append(loss_custom)
            custom_batch_acc.append(custom_acc)
        custom_valid_avg_loss = torch.tensor(custom_batch_loss).mean()
        custom_valid_avg_acc  = torch.tensor(custom_batch_acc).mean()
        if epoch == 1: 
            torch.save(model_custom.state_dict(), result_path + '/ex5_weights.pth')
            best_loss = custom_valid_avg_loss
        if best_loss > custom_valid_avg_loss:
            best_loss = custom_valid_avg_loss
            torch.save(model_custom.state_dict(), result_path + '/ex5_weights.pth')
        

        # custom_scheduler.step(custom_valid_avg_loss)
    # 学習過程を記録
    history['custom_train_loss'].append(custom_train_avg_loss)
    history['custom_valid_loss'].append(custom_valid_avg_loss)
    history['custom_train_acc'].append(custom_train_avg_acc)
    history['custom_valid_acc'].append(custom_valid_avg_acc)
    
    logging.info(f"| Custom_Train | Epoch   {epoch+1} |: custom_train_loss:{custom_train_avg_loss:.3f}, custom_train_acc:{custom_train_avg_acc*100:3.3f}% | custom_valid_loss:{custom_valid_avg_loss:.5f}, custom_valid_acc:{custom_valid_avg_acc*100:3.3f}%")
    print(f"| Custom_Train | Epoch   {epoch+1} |: custom_train_loss:{custom_train_avg_loss:.3f}, custom_train_acc:{custom_train_avg_acc*100:3.3f}% | custom_valid_loss:{custom_valid_avg_loss:.5f}, custom_valid_acc:{custom_valid_avg_acc*100:3.3f}%")

filename_acc = log_img_path+ "/custom_Accuracy.png"
filename_loss = log_img_path+ "/custom_loss.png"
fig1 = plt.figure()
plt.plot(history['custom_train_acc'])
plt.plot(history['custom_valid_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['custom train acc', 'custom validation acc'], loc='lower right')
plt.grid(True)
fig1.savefig(filename_acc)
# Lossを描画
fig2 = plt.figure()
plt.plot(history['custom_train_loss'])
plt.plot(history['custom_valid_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['custom train loss', 'custom validation loss'], loc='upper right')
plt.grid(True)
fig2.savefig(filename_loss)


custom_batch_loss = []
custom_batch_acc = []
pred = np.array([])
true = np.array([])

model_custom = models.resnet18(pretrained=False,num_classes=args.num_classes)                                    # 保存時と全く同じ構成のモデルを構築
params = torch.load(result_path + '/ex5_weights.pth', map_location=device)   # 保存した重みを読み出す
model_custom.load_state_dict(params) 
model_custom.to(device)
model_custom.eval()

# 勾配計算に必要な情報を記録しないよう設定
count=0
with torch.no_grad():
    loop = tqdm(test_loader, unit='batch', desc='| Test | Epoch {:>3} |'.format(epochs+1))
    for batch in loop:
        count+=1
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        out_custom = model_custom(x)
        custom_pred_class = torch.argmax(out_custom, dim=1)
        acc_custom = torch.sum(custom_pred_class == label)/len(label)
        custom_batch_acc.append(acc_custom)

        pred = np.append(pred,custom_pred_class.to('cpu').detach().numpy().copy())
        true = np.append(true,label.to('cpu').detach().numpy().copy())

    custom_test_avg_loss = torch.tensor(custom_batch_loss).mean()
    custom_test_avg_acc  = torch.tensor(custom_batch_acc).mean()

print(f'precision:{precision_score(true,pred,zero_division=True)}')
print(f'recall:{recall_score(true,pred)}')
print(f'f1:{f1_score(true,pred)}')


print(f"custom_test_loss   ：{custom_test_avg_loss:3.5f}")
print(f"custom_test_acc    ：{custom_test_avg_acc*100:2.3f}%")
logging.info(f"custom_test_loss   ：{custom_test_avg_loss:3.5f}")
logging.info(f"custom_test_acc    ：{custom_test_avg_acc*100:2.3f}%")
logging.info(f'precision:{precision_score(true,pred,zero_division=True)}')
logging.info(f'recall:{recall_score(true,pred)}')
logging.info(f'f1:{f1_score(true,pred)}')
