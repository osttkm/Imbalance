from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
from glob import glob
from pathlib import Path
import os
import random
# from .sampler import BalancedBatchSampler
import json

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class shoden_dataset(Dataset): 
    def __init__(self,seed,mode,transform,num_samples=0,paths=[],add_center=0): 
        self.mode=mode
        self.transform=transform
        self.seed = seed
        """シード値の固定"""
        # self.random_seed = seed
        # random.seed(self.random_seed)
        # np.random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(self.random_seed)
        #     torch.backends.cudnn.deterministic = True


        self.train_imgs={
            "data":[],
            "label":[],
            "index":[],
        }
        self.valid_imgs={
            "data":[],
            "label":[],
            "index":[],
        }
        self.test_imgs={
            "data":[],
            "label":[],
            "index":[],
        }
        self.rb_imgs={
            "data":[],
            "label":[],
            "index":[],
        }
        self.cb_imgs={
            "data":[],
            "label":[],
        }

        data_path = Path('/home/dataset/shoden/shoden_multi+ok/training/')
        data_dir = [path for path in data_path.iterdir() if path.is_dir()]
    
        json_open = open('/home/oshita/cleansing/my_project/shoden.json', 'r')
        data = json.load(json_open)
        for idx,d in zip(range(len(data)),data):
            for patch in d['patches']:
                ### test images(base image 23~41 & 95~100)
                if patch['category_index']==2 or patch['category_index']==3:
                    if patch['category_index']==2: 
                        self.test_imgs['data'].append(str(data_dir[2])+'/'+patch['name'])
                        self.test_imgs['label'].append(torch.tensor(0))
                    elif patch['category_index']==3: 
                        self.test_imgs['data'].append(str(data_dir[3])+'/'+patch['name'])
                        self.test_imgs['label'].append(torch.tensor(1))
                    else: raise ValueError('miss action is occur during making test dataset')
                    self.test_imgs['index'].append(idx)

                ### train and valid images(base image 0~22 &42~94)
                else:
                    val_index=np.array([3,5,8,10,11,15,17,46,81,92,93])
                    if np.sum(val_index == idx)==0:
                        if patch['category_index']==0: 
                            self.train_imgs['data'].append(str(data_dir[0])+'/'+patch['name'])
                            self.train_imgs['label'].append(torch.tensor(0))
                        elif patch['category_index']==1: 
                            self.train_imgs['data'].append(str(data_dir[1])+'/'+patch['name'])
                            self.train_imgs['label'].append(torch.tensor(1))
                        else: raise ValueError('miss action is occur during making train dataset')
                        self.train_imgs['index'].append(idx)
                    elif np.sum(val_index == idx)!=0:
                        if patch['category_index']==0: 
                            self.valid_imgs['data'].append(str(data_dir[0])+'/'+patch['name'])
                            self.valid_imgs['label'].append(torch.tensor(0))
                        elif patch['category_index']==1: 
                            self.valid_imgs['data'].append(str(data_dir[1])+'/'+patch['name'])
                            self.valid_imgs['label'].append(torch.tensor(1))
                        else: raise ValueError('miss action is occur during making valid dataset')
                        self.valid_imgs['index'].append(idx)
        random.seed(43)
        if self.mode=='rb':
            """balanced dataset"""
            anormal_path = np.array(self.train_imgs['data'])[np.array(self.train_imgs['label'])==1]
            index = np.array(self.train_imgs['index'])[np.array(self.train_imgs['label'])==1]
            for i,path,idx in zip(range(len(anormal_path)),anormal_path,index):
                self.rb_imgs['data'].append(path)
                self.rb_imgs['label'].append(torch.tensor(1)) 
                self.rb_imgs['index'].append(idx)
            "ngにデータ数を合わせる"
            normal_path=np.array(self.train_imgs['data'])[np.array(self.train_imgs['label'])==0]
            index=np.array(self.train_imgs['index'])[np.array(self.train_imgs['label'])==0]
            normal_path = random.sample(normal_path.tolist(),add_center+len(anormal_path))
            index = random.sample(index.tolist(),add_center+len(anormal_path))
            for i,path,idx in zip(range(len(normal_path)),normal_path,index):
                self.rb_imgs['data'].append(path)
                self.rb_imgs['label'].append(torch.tensor(0))
                self.rb_imgs['index'].append(idx)
        random.seed(self.seed)
            
     
           
        
        if self.mode=='cb':
            for path in paths:
                self.cb_imgs['data'].append(path)
                path = Path(path)
                # import pdb;pdb.set_trace()
                if path.parent.name.split('_')[1] == 'OK':self.cb_imgs['label'].append(0)
                elif path.parent.name.split('_')[1] == 'NG': self.cb_imgs['label'].append(1)
                else:
                    raise ValueError('invalid name file may exist')



    def __getitem__(self, index):
        if self.mode=='train':
            img = Image.open(self.train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.train_imgs["label"][index],self.train_imgs["data"][index]
        elif self.mode=='valid':
            img = Image.open(self.valid_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.valid_imgs["label"][index],self.valid_imgs["data"][index]
        elif self.mode=='rb':
            img = Image.open(self.rb_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.rb_imgs["label"][index],self.rb_imgs["data"][index]
        elif self.mode=='cb':
            img = Image.open(self.cb_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.cb_imgs["label"][index],self.cb_imgs["data"][index]
        else:
            img = Image.open(self.test_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.test_imgs["label"][index],self.test_imgs["data"][index]
            
        
    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs["data"])
        if self.mode=="valid":
            return len(self.valid_imgs["data"])
        if self.mode=="rb":
            return len(self.rb_imgs["data"])
        if self.mode=="cb":
            return len(self.cb_imgs["data"])
        else:
            return len(self.test_imgs["data"])



class shoden_loader():
    def __init__(self,args):
        """シード値の固定"""
        self.random_seed = args.seed
        # random.seed(self.random_seed)
        # np.random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(self.random_seed)
        #     torch.backends.cudnn.deterministic = True

        self.transform = transforms.Compose([
                transforms.ToTensor(),       
                transforms.RandomRotation(90),         
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)), ])     

        self.transform_test = transforms.Compose([
                transforms.ToTensor(),               
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),  ])
            
        self.train_dataset=shoden_dataset(self.random_seed,'train',self.transform,num_samples=50000)
        self.valid_dataset=shoden_dataset(self.random_seed,'valid',self.transform,num_samples=50000)
        self.test_dataset=shoden_dataset(self.random_seed,'test',self.transform,num_samples=50000)

    
    def run(self,mode,path=[],add_center=0):
        if mode=='train':
            train_loader = DataLoader(
                self.train_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=256,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return train_loader
        
        elif mode=='valid':
            valid_loader = DataLoader(
                self.valid_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=32,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return valid_loader
        # elif mode=='sampler':
        #     sampled_dataset = shoden_dataset(self.random_seed,'train',self.transform,num_samples=50000)
        #     sampled_loader = DataLoader(
        #         sampled_dataset,
        #         pin_memory=True,
        #         batch_sampler=BalancedBatchSampler(sampled_dataset,2,128),
        #         num_workers=os.cpu_count()
        #     )
        #     return sampled_loader
        elif mode=='test':
            test_loader = DataLoader(
                self.test_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=32,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return test_loader
        elif mode=='rb':
            rb_dataset = shoden_dataset(self.random_seed,'rb',self.transform,num_samples=50000,paths=path,add_center=add_center)
            rb_loader = DataLoader(
                rb_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=32,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return rb_loader
        elif mode=='cb':
            cb_dataset = shoden_dataset(self.random_seed,'cb',self.transform,num_samples=50000,paths=path)
            cb_loader = DataLoader(
                cb_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=32,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return cb_loader
    def get_num(self,name,loader=None):
        if loader!=None:
            cls_1,cls_2=0,0
            for index,(data,label,_) in enumerate(loader):
                data=data.to(device)
                label=label.to(device)
                cls_1+=len(label[label==0])
                cls_2+=len(label[label==1])
            return [cls_1,cls_2]

        else:
            data = self.run(name)
            cls_1,cls_2=0,0
            for index,(data,label,_) in enumerate(data):
                data=data.to(device)
                label=label.to(device)
                cls_1+=len(label[label==0])
                cls_2+=len(label[label==1])
            return [cls_1,cls_2]
        

       
if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='ハイパラに関して')
    parser.add_argument('--seed',type=int,default=9999)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    loader=shoden_loader(args)
    # tr_list=loader.get_num('train')
    # va_list=loader.get_num('valid')
    te_list=loader.get_num('test')
    # print(f'train:{tr_list}')
    # print(f'valid:{va_list}')
    print(f'test:{te_list}')
    # import pdb;pdb.set_trace()
   
    