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
result_path = f"./{path[2:9]}_{path[10:13]}_rand_result_aug:{args.aug}_seed:{args.seed}"
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
rand_train_loader=loader.run('random')

model_rand=models.resnet18(pretrained=False,num_classes=args.num_classes).to(device)
gpu_num = torch.cuda.device_count()
if(torch.cuda.device_count()>1): 
    model_rand = torch.nn.DataParallel(model_rand,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())

opt_rand=optim.SGD(model_rand.parameters(),lr=0.1,weight_decay=0.001)

# rand_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_rand,patience=5,factor=0.1)
rand_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_rand,T_max=10,eta_min=0.0001)
criterion = nn.CrossEntropyLoss()

history={
    'rand_train_loss':[],
    'rand_train_acc':[],
    'rand_valid_loss':[],
    'rand_valid_acc':[]
}

epochs = args.epoch
best_loss = 0
for epoch in range(epochs):

    """randomで選んだ学習データの訓練"""
    loop_rand = tqdm(rand_train_loader, unit='batch', desc='| Random Train | Epoch {:>3} |'.format(epoch+1))
    model_rand.train()
    scaler_rand = torch.cuda.amp.GradScaler()
    batch_loss = []
    batch_acc  = []
    for batch in loop_rand:  
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # import pdb;pdb.set_trace()
        
        opt_rand.zero_grad()       
        with torch.cuda.amp.autocast():
            out = model_rand(x)             
            loss = criterion(out, label)    
        scaler_rand.scale(loss).backward()
        scaler_rand.step(opt_rand)
        scaler_rand.update()
       
        pred_class = torch.argmax(out, dim=1)       
        acc = torch.sum(pred_class == label)/len(label) 

        batch_loss.append(loss)
        batch_acc.append(acc)

    rand_train_avg_loss = torch.tensor(batch_loss).mean()
    rand_train_avg_acc  = torch.tensor(batch_acc).mean()


    #========== 検証用データへの処理 ==========#
    # モデルを評価モードに設定
    model_rand.eval()
    rand_batch_loss = []
    rand_batch_acc = []

    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in valid_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 順伝播と誤差の計算のみ行う
            out_rand = model_rand(x)
            loss_rand = criterion(out_rand, label)

            # 正答率(accuracy)を算出
            rand_pred_class = torch.argmax(out_rand,dim=1)
            rand_acc = torch.sum(rand_pred_class == label)/len(label)

            # 1バッチ分の誤差をリストに保存
            
            rand_batch_loss.append(loss_rand)
            rand_batch_acc.append(rand_acc)
            

        # バッチ単位の結果を平均し、1エポック分の誤差を算出
        
        rand_valid_avg_loss = torch.tensor(rand_batch_loss).mean()
        rand_valid_avg_acc  = torch.tensor(rand_batch_acc).mean()
        if epoch == 1: 
            torch.save(model_rand.state_dict(), result_path + '/ex5_weights.pth')
            best_loss = rand_valid_avg_loss
        if best_loss > rand_valid_avg_loss:
            best_loss = rand_valid_avg_loss
            torch.save(model_rand.state_dict(), result_path + '/ex5_weights.pth')
        

    # 学習過程を記録
    history['rand_train_loss'].append(rand_train_avg_loss)
    history['rand_valid_loss'].append(rand_valid_avg_loss)
    history['rand_train_acc'].append(rand_train_avg_acc)
    history['rand_valid_acc'].append(rand_valid_avg_acc)
    
    logging.info(f"| Rand_Train | Epoch   {epoch+1} |: rand_train_loss:{rand_train_avg_loss:.3f}, rand_train_acc:{rand_train_avg_acc*100:3.3f}% | rand_valid_loss:{rand_valid_avg_loss:.5f}, rand_valid_acc:{rand_valid_avg_acc*100:3.3f}%")
    print(f"| Rand_Train | Epoch   {epoch+1} |: rand_train_loss:{rand_train_avg_loss:.3f}, rand_train_acc:{rand_train_avg_acc*100:3.3f}% | rand_valid_loss:{rand_valid_avg_loss:.5f}, rand_valid_acc:{rand_valid_avg_acc*100:3.3f}%")



rand_filename_acc = log_img_path+ "/rand_Accuracy.png"
rand_filename_loss = log_img_path+ "/rand_loss.png"
fig5 = plt.figure()
plt.plot(history['rand_train_acc'])
plt.plot(history['rand_valid_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['rand train acc', 'rand validation acc'], loc='lower right')
plt.grid(True)
fig5.savefig(rand_filename_acc)
# Lossを描画
fig6 = plt.figure()
plt.plot(history['rand_train_loss'])
plt.plot(history['rand_valid_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['rand train loss', 'rand validation loss'], loc='upper right')
plt.grid(True)
fig6.savefig(rand_filename_loss)

rand_batch_loss = []
rand_batch_acc = []
pred = np.array([])
true = np.array([])

model_rand = models.resnet18(pretrained=False,num_classes=args.num_classes)                                    # 保存時と全く同じ構成のモデルを構築
params = torch.load(result_path + '/ex5_weights.pth', map_location=device)   # 保存した重みを読み出す
model_rand.load_state_dict(params) 
model_rand.to(device)
model_rand.eval()

# 勾配計算に必要な情報を記録しないよう設定
with torch.no_grad():
    loop = tqdm(test_loader, unit='batch', desc='| Test | Epoch {:>3} |'.format(epochs+1))
    for batch in loop:
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        out_rand = model_rand(x)
        rand_pred_class = torch.argmax(out_rand, dim=1)

        acc_rand = torch.sum(rand_pred_class == label)/len(label)
        rand_batch_acc.append(acc_rand)

        pred = np.append(pred,rand_pred_class.to('cpu').detach().numpy().copy())
        true = np.append(true,label.to('cpu').detach().numpy().copy())

    rand_test_avg_loss = torch.tensor(rand_batch_loss).mean()
    rand_test_avg_acc  = torch.tensor(rand_batch_acc).mean()

print(f'precision:{precision_score(true,pred,zero_division=True)}')
print(f'recall:{recall_score(true,pred)}')
print(f'f1:{f1_score(true,pred)}')
print(f"rand_test_loss   ：{rand_test_avg_loss:3.5f}")
print(f"rand_test_acc    ：{rand_test_avg_acc*100:2.3f}%")
logging.info(f"rand_test_loss   ：{rand_test_avg_loss:3.5f}")
logging.info(f"rand_test_acc    ：{rand_test_avg_acc*100:2.3f}%")
logging.info(f'precision:{precision_score(true,pred,zero_division=True)}')
logging.info(f'recall:{recall_score(true,pred)}')
logging.info(f'f1:{f1_score(true,pred)}')
