import argparse
import random
import csv
import os
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.coresetv2 import furthest_first as coresetv2
from src.normal_coreset import furthest_first as normal_coreset
from src.util import *
from src.shoden_loader import shoden_loader
from src.mnist_loader import mnist_loader
from src.cifar_loader import cifar_loader
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve,auc
# その他
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.RSG import * 
import src.MlflowWriter as MW
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER
from src.losses import *
from PIL import Image
from resnet import resnet as my_resnet
from resnet import ssl_resnet as my_ssl_resnet
from model_second import Classifier

device = torch.device('cuda')
model_path = './cb_models'
os.makedirs(model_path,exist_ok=True)

parser = argparse.ArgumentParser(description='ハイパラに関して')
parser.add_argument('--dataset', help = 'use data',type=str,default='mnist',choices=['mnist','shoden','cifar'])
parser.add_argument('--model', help = 'arch',type=str,default='resnet18',choices=['my_model','resnet18','ssl_resnet18'])
parser.add_argument('--extractor', help = 'arch',type=str,default='resnet18',choices=['resnet18','ssl_resnet18'])
parser.add_argument('--epoch', help = 'epoch',type=int,default=150)
parser.add_argument('--lr', help= 'learning rate',type=float,default=0.01)
parser.add_argument('--num_classes',type=int,default=2)
parser.add_argument('--weight_decay',type=float,default=0.0001)
parser.add_argument('--mode',type=str,default='rb',choices=['rb','cb','crb','org','v2'])
parser.add_argument('--seed',type=int,default=9999)
parser.add_argument('--save_flag',type=str,default='True')
parser.add_argument('--tag',type=str,default='no description')
parser.add_argument('--add_center',type=int,default=0)
parser.add_argument('--layer',type=int,default=2)
parser.add_argument('--coreset',type=str,default='coreset')
parser.add_argument('--setting',type=int,default=1)
parser.add_argument('--maj',type=int,default=0)
parser.add_argument('--min',type=int,default=5)
parser.add_argument('--min_class_num',type=int,default=200)

args = parser.parse_args()

EXPERIMENT_NAME = f'{args.tag+str(args.min_class_num)}'
writer = MW.MlflowWriter(EXPERIMENT_NAME)
tags = {'trial':args.seed,
        'epoch':args.epoch,
        MLFLOW_RUN_NAME:f'tag:{args.dataset}_{args.mode}_{args.add_center}_{args.seed}_{args.setting}',
        'tag':args.tag,
        MLFLOW_USER:args.model}
writer.create_new_run(tags)
writer.log_param('mode',args.mode)
writer.log_param('epoch',args.epoch)
writer.log_param('setting',args.setting)
if args.mode=='cb' or args.mode=='v2':
    writer.log_param('extractor',args.mode)
    writer.log_param('layer',args.layer)
    writer.log_param('add_center',args.add_center)
    writer.log_param('coreset',args.coreset)
if args.mode=='rb':
    writer.log_param('add_center',args.add_center)

"""### シード値の固定"""
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

if args.dataset == 'shoden':
    loader=shoden_loader(args)
elif args.dataset=='mnist':
    loader = mnist_loader()
elif args.dataset=='cifar':
    loader = cifar_loader(args)

if args.mode == 'rb':
    train = loader.run('rb',add_center=args.add_center)
    valid,test = loader.run('valid'),loader.run('test')

elif args.mode == 'org': 
    train,valid,test = loader.run('train'),loader.run('valid'),loader.run('test')
    all_path = []
    all_label = torch.tensor([]).to(device)
    with torch.no_grad():
        for _,batch in enumerate(train):
            data,label,path=batch
            all_path.extend(path)
    all_path = np.array(all_path)
    train = loader.run('cb',all_path)

elif args.mode == 'cb':
    train_data = loader.run('train')
    if args.extractor == 'resnet18':
        model = my_resnet(layer=args.layer, coreset=args.coreset)
    elif args.extractor == 'ssl_resnet18':
        device = torch.device('cuda')
        model = my_ssl_resnet(layer=args.layer, coreset=args.coreset)
    model.to(device)
    model.eval()
    feature = torch.tensor([]).to(device)
    all_target = np.array([])
    all_path = [];train_path = []
    with torch.no_grad():
        for _,batch in enumerate(train_data):
            data,label,path=batch
            data,label=data.to(device),label.to(device)
            if args.coreset=='coreset': 
                feat = model(data)
            elif args.coreset=='greedy_coreset':
                feat = model(data)
            feature = torch.cat((feature,feat),dim=0)
            all_path.extend(path)
            all_target = np.append(all_target,label.to('cpu').detach().numpy().copy())
    all_path = np.array(all_path)
    normal_path = np.array(all_path[all_target==0])
    anormal_path = np.array(all_path[all_target==1])
    init_num=100
    print('normal coreset')
    idx=normal_coreset(feature[all_target==0],False,len(all_target[all_target==1])+args.add_center,init_num=init_num)
    path = np.append(normal_path[idx],anormal_path)
    with open("./text.txt", 'w') as f:
        for p in path:
            f.write(p+'\n') 
    writer.log_artifact("./text.txt")  
    os.remove("./text.txt")
    # import pdb;pdb.set_trace()
    train = loader.run('cb',path)
    valid,test = loader.run('valid'),loader.run('test')

# elif args.mode == 'crb':
#     train_data = loader.run('train')
#     if args.extractor == 'resnet18':
#         model = my_resnet(layer=args.layer, coreset=args.coreset)
#     model.to(device)
#     model.eval()
#     feature = torch.tensor([]).to(device)
#     all_target = np.array([])
#     all_path = [];train_path = []
#     with torch.no_grad():
#         for _,batch in enumerate(train_data):
#             data,label,path=batch
#             data,label=data.to(device),label.to(device)
#             feat = model(data)
#             feature = torch.cat((feature,feat),dim=0)
#             all_path.extend(path)
#             all_target = np.append(all_target,label.to('cpu').detach().numpy().copy())
#     all_path = np.array(all_path)
#     normal_path = np.array(all_path[all_target==0])
#     anormal_path = np.array(all_path[all_target==1])
#     init_num=100
#     # if args.tag == 'normal_coreset_layer1' or args.tag == 'normal_coreset_layer2' or args.tag == 'normal_coreset_layer3': 
#     #     print('normal coreset')
#     #     idx=normal_coreset(feature[all_target==0],False,len(all_target[all_target==1])+args.add_center,init_num=init_num,seed=args.seed)
#     idx = normal_coreset(feature[all_target==0],False,len(all_target[all_target==1])+args.turning,init_num=init_num)
#     path = np.append(normal_path[idx],anormal_path)
#     for del_path in normal_path[idx]:
#         normal_path = np.delete(normal_path,np.argwhere(normal_path==del_path))
#     random_path = random.sample(normal_path.tolist(),args.add_center - args.turning)
#     path = np.append(path,random_path)
#     with open("./text.txt", 'w') as f:
#         for p in path:
#             f.write(p+'\n') 
#     writer.log_artifact("./text.txt")  
#     os.remove("./text.txt")
#     train = loader.run('cb',path)
#     valid,test = loader.run('valid'),loader.run('test')

elif args.mode == 'v2':
    train_data = loader.run('train')
    if args.extractor == 'resnet18':
        model = my_resnet(layer=args.layer, coreset=args.coreset)
    model.to(device)
    model.eval()
    feature = torch.tensor([]).to(device)
    all_target = np.array([])
    all_path = [];train_path = []
    with torch.no_grad():
        for _,batch in enumerate(train_data):
            data,label,path=batch
            data,label=data.to(device),label.to(device)
            feat = model(data)
            feature = torch.cat((feature,feat),dim=0)
            all_path.extend(path)
            all_target = np.append(all_target,label.to('cpu').detach().numpy().copy())
    all_path = np.array(all_path)
    normal_path = np.array(all_path[all_target==0])
    anormal_path = np.array(all_path[all_target==1])
    init_num=100
    idx = coresetv2(feature[all_target==0],False,len(all_target[all_target==1])+args.add_center,init_num=init_num)
    path = np.append(normal_path[idx],anormal_path)
    with open("./text.txt", 'w') as f:
        for p in path:
            f.write(p+'\n') 
    writer.log_artifact("./text.txt")  
    os.remove("./text.txt")
    train = loader.run('cb',path)
    valid,test = loader.run('valid'),loader.run('test')
else:
    raise ValueError('invalid mode is selected')

"""### シード値の固定"""
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


if args.model=='ssl_resnet18':
    model=models.resnet18(pretrained=False,num_classes=args.num_classes)
    param=torch.load('/home/oshita/cleansing/my_project/simsiam/checkpoint/checkpoint_ver.pth.tar', map_location=device)  
    model.load_state_dict(param['state_dict'],strict=False) 
elif args.model=='resnet18':
    model = models.resnet18(pretrained=False,num_classes=args.num_classes)
elif args.model=='my_model':
    model = Classifier(enc_dim=128,num_classes=args.num_classes)
    
model.apply(initialize_weights)
model.to(device)
gpu_num = torch.cuda.device_count()
if(torch.cuda.device_count()>1): 
    model = torch.nn.DataParallel(model,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())

criterion = nn.CrossEntropyLoss(weight=None).to(device)
if args.setting==1:
    optimizer = optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    _scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=0.0001,T_max=50)
elif args.setting==2:
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    _scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.95)
else:
    raise ValueError('invalid train setting')

import pdb;pdb.set_trace()
"""### モデルの学習"""
epochs = args.epoch
best_loss=0
# エポックの数だけ繰り返す
for epoch in range(epochs):
    loop = tqdm(train, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    batch_loss = []
    batch_acc  = []
    # ミニバッチ単位で繰り返す
    true_label,pred_label=torch.tensor([]).to(device),torch.tensor([]).to(device)
    for batch in loop:  
        x, label,_ = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # 学習のメイン処理
        optimizer.zero_grad()       # (1) パラメータの勾配を初期化
        with torch.cuda.amp.autocast():
            out = model(x)              # (2) データをモデルに入力(順伝播)
            loss = criterion(out, label)    # (3) 誤差関数の値を算出
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 正答率(accuracy)を算出
        pred_class = torch.argmax(out, dim=1)        # モデルの出力から予測ラベルを算出
        acc = torch.sum(pred_class == label)/len(label)
        true_label = torch.cat((true_label,label),dim=0)
        pred_label = torch.cat((pred_label,pred_class),dim=0)

        # 1バッチ分の誤差と正答率をリストに保存
        batch_loss.append(loss)
        batch_acc.append(acc)

    # バッチ単位の結果を平均し、1エポック分の誤差を算出
    train_avg_loss = torch.tensor(batch_loss).mean()
    train_avg_acc  = torch.tensor(batch_acc).mean()
    true_label = true_label.to('cpu').detach().numpy().copy()
    pred_label = pred_label.to('cpu').detach().numpy().copy()
    writer.log_metric_step('train_loss',train_avg_loss,epoch)
    writer.log_metric_step('train_acc',train_avg_acc,epoch)
    writer.log_metric_step('train_recall',np.round(recall_score(true_label,pred_label),5),epoch)
    writer.log_metric_step('train_precision',np.round(precision_score(true_label,pred_label,zero_division=True),5),epoch)
    writer.log_metric_step('train_f1_score',np.round(f1_score(true_label,pred_label),5),epoch)
    #========== 検証用データへの処理 ==========#
    # モデルを評価モードに設定
    model.eval()
    batch_loss = []
    batch_acc = []
    true_label,pred_label=torch.tensor([]).to(device),torch.tensor([]).to(device)
    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in valid:
            x, label,_ = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 順伝播と誤差の計算のみ行う
            out = model(x)              # (2) データをモデルに入力(順伝播)
            loss = criterion(out, label)    # (3) 誤差関数の値を算出
           
            # out = get_output('eval',x,label,epoch,model)
            # loss = criterion(out, label)
            # 正答率(accuracy)を算出
            pred_class = torch.argmax(out, dim=1)
            acc = torch.sum(pred_class == label)/len(label)
            true_label = torch.cat((true_label,label),dim=0)
            pred_label = torch.cat((pred_label,pred_class),dim=0)

            # 1バッチ分の誤差をリストに保存
            batch_loss.append(loss)
            batch_acc.append(acc)
            # schedulerがReduceLRなのでval_losswp渡さないといけないらしい
            # そのため検証時に更新
            

        # バッチ単位の結果を平均し、1エポック分の誤差を算出
        valid_avg_loss = torch.tensor(batch_loss).mean()
        valid_avg_acc  = torch.tensor(batch_acc).mean()
        true_label = true_label.to('cpu').detach().numpy().copy()
        pred_label = pred_label.to('cpu').detach().numpy().copy()
        writer.log_metric_step('valid_loss',valid_avg_loss,epoch)
        writer.log_metric_step('valid_acc',valid_avg_acc,epoch)
        writer.log_metric_step('valid_recall',np.round(recall_score(true_label,pred_label),5),epoch)
        writer.log_metric_step('valid_precision',np.round(precision_score(true_label,pred_label,zero_division=True),5),epoch)
        writer.log_metric_step('valid_f1_score',np.round(f1_score(true_label,pred_label),5),epoch)
        if epoch == 1: 
            print('initial init')
            torch.save(model.state_dict(), model_path +'/'+tags[MLFLOW_RUN_NAME] +'.pth')
            best_loss = valid_avg_loss
        if best_loss > valid_avg_loss:
            print('update best model')
            best_loss = valid_avg_loss
            torch.save(model.state_dict(), model_path + '/'+tags[MLFLOW_RUN_NAME]+'.pth')
    
    print(f"| Train | Epoch   {epoch+1} |: train_loss:{train_avg_loss:.3f}, train_acc:{train_avg_acc*100:3.3f}% | valid_loss:{valid_avg_loss:.5f}, valid_acc:{valid_avg_acc*100:3.3f}%")
print('Finished Training')



if args.model == 'resnet18': model = models.resnet18(pretrained=False,num_classes=args.num_classes)   
elif args.model == 'ssl_resnet18': model = models.resnet18(pretrained=False,num_classes=args.num_classes) 
elif args.model == 'my_model': model = Classifier(enc_dim=128,num_classes=args.num_classes) 
else: raise ValueError('no model are exist!!')                                # 保存時と全く同じ構成のモデルを構築
params = torch.load(model_path +'/'+tags[MLFLOW_RUN_NAME] +'.pth', map_location=device)   # 保存した重みを読み出す
model.load_state_dict(params,strict=True) 
model.to(device)
model.eval()

batch_loss = []
batch_acc = []
true_label=torch.tensor([]).to(device)
pred_label=torch.tensor([]).to(device)
save_label=torch.tensor([]).to(device)
save_path=np.array([])
pred=np.array([])
# クラスごとの正答率も残しておく
class_correct = [0,0]
class_total = [0,0]

with torch.no_grad():
    loop = tqdm(test, unit='batch', desc='| Test | Epoch {:>3} |'.format(1))
    count=-1
    softmax = nn.Softmax(dim=1)
    # ミニバッチ単位で繰り返す
    for batch in loop:
        count+=1
        x, label,path = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        out = model(x)            
        pred=np.append(pred,softmax(out)[:,1].to('cpu').detach().numpy().copy())
 
        pred_class = torch.argmax(out, dim=1)
        acc = torch.sum(pred_class == label)/len(label)
        true_label = torch.cat((true_label,label),dim=0)
        pred_label = torch.cat((pred_label,pred_class),dim=0)

        for predict,l in zip(pred_class,label):
            if predict==l: class_correct[l] += 1
            class_total[l] += 1

        if args.save_flag == 'True':
            for l,_pred,p in zip(label,pred_class,path):
                if (l-_pred)!=0:
                    name=''
                    img = Image.open(p)
                    if l-_pred==-1: name=f'over_detect_{softmax(out)[0][_pred]}:{count}.png'
                    elif l-_pred==1: name=f'miss_defect_{softmax(out)[0][_pred]}:{count}.png'
                    img.save('./'+name)
                    writer.log_artifact('./'+name)
                    os.remove('./'+name)
        batch_acc.append(acc)

    # バッチ単位の結果を平均し、1エポック分の誤差を算出
    test_avg_loss = torch.tensor(batch_loss).mean()
    test_avg_acc  = torch.tensor(batch_acc).mean()

for i in range(args.num_classes):
    _class = ['normal','anormal']
    print(f'{[_class[i]]} class:{class_correct[i] / class_total[i]}%')
    writer.log_metric(f'{_class[i]}_acc',class_correct[i] / class_total[i])
print("テストデータに対する結果")
print(f"test_loss   ：{test_avg_loss:3.5f}")
print(f"test_acc    ：{test_avg_acc*100:2.3f}%")

true_label = true_label.to('cpu').detach().numpy().copy()
pred_label = pred_label.to('cpu').detach().numpy().copy()

writer.log_metric('test_acc',test_avg_acc)
writer.log_metric('test_recall',recall_score(true_label,pred_label))
writer.log_metric('test_precision',precision_score(true_label,pred_label,zero_division=True))
writer.log_metric('test_f1_score',f1_score(true_label,pred_label))
precision, recall, threshold = precision_recall_curve(true_label, pred)


data=[{
    "PRAUC":auc(recall,precision),
    "normal_acc":class_correct[0] / class_total[0],
    "anormal_acc":class_correct[1] / class_total[1],
    "test_acc":test_avg_acc.item(),
    "test_recall":recall_score(true_label,pred_label),
    "test_precision":precision_score(true_label,pred_label,zero_division=True),
    "test_f1_score":f1_score(true_label,pred_label),
    "run_name":args.tag,
    "seed":args.seed,
    "add_center":args.add_center,
}]
if not os.path.isfile(f'./log/{args.tag}.csv'):
    with open(f'./log/{args.tag}.csv', "w", newline="") as f:
        header_row=["PRAUC","normal_acc","anormal_acc","test_acc", "test_recall", "test_precision","test_f1_score","run_name","seed","add_center"]
        csv_writer = csv.writer(f)
        csv_writer.writerow(header_row)
with open(f'./log/{args.tag}.csv','a',newline='') as f:
    fieldnames = ["PRAUC","normal_acc","anormal_acc","test_acc", "test_recall", "test_precision","test_f1_score","run_name","seed","add_center"]
    dict_writer = csv.DictWriter(f, fieldnames=fieldnames,extrasaction='ignore')
    # dict_writer.writeheader()
    dict_writer.writerows(data)


from matplotlib import font_manager
import matplotlib
font_files = font_manager.findSystemFonts(fontpaths=["./fonts"])
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
matplotlib.rc('font', family="Times New Roman")
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定(山田先輩もこのstixの設定だった)
plt.rcParams["font.size"] = 10.5 # 全体のフォントサイズが変更されます。


fig, ax = plt.subplots(facecolor="w", figsize=(5, 5))
ax.grid()
ax.plot(precision, recall)
ax.set_xlabel("Precision")
ax.set_ylabel("Recall")
plt.title(f'auc:{auc(recall,precision):.4f}')
plt.savefig('./pr.png')
writer.log_artifact('./pr.png')
writer.log_metric('PRAUC',auc(recall,precision))
print(f'PRAUC:{auc(recall,precision)}')
os.remove('./pr.png')
os.remove(model_path +'/'+tags[MLFLOW_RUN_NAME] +'.pth')
writer.set_terminated()
print('Experiment is all finished')


