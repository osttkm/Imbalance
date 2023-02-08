from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os
import random


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class cifar_dataset(Dataset): 
    def __init__(self,args,mode,transform,num_samples=0,paths=[],add_center=0): 
        self.args=args
        self.mode=mode
        self.transform=transform

        self.train_imgs={
            "data":[],
            "label":[],
        }
        self.valid_imgs={
            "data":[],
            "label":[],
        }
        self.test_imgs={
            "data":[],
            "label":[],
        }
        self.rb_imgs={
            "data":[],
            "label":[],
        }
        self.cb_imgs={
            "data":[],
            "label":[],
        }

        # サンプリングの固定
        random.seed(43)
        data_path = Path('/home/dataset/cifar10')
        data_dir = [path for path in data_path.iterdir() if path.is_dir()]
        for dir in data_dir:
            if dir.name=='train':
                maj_list = list(dir.glob(str(self.args.maj)+'/*.png'))
                min_list = list(dir.glob(str(self.args.min)+'/*.png'))
                
                _train_min_sample = random.sample(min_list,self.args.min_class_num+200)
                random.shuffle(maj_list)
                train_maj_sample = maj_list[0:4000]
                train_min_sample = _train_min_sample[0:args.min_class_num]
                valid_maj_sample = maj_list[4000:]
                valid_min_sample = _train_min_sample[args.min_class_num:]
                for path in train_maj_sample+train_min_sample:
                    self.train_imgs['data'].append(str(path))
                    if int(path.parent.name)==args.maj:
                        self.train_imgs['label'].append(torch.tensor(0))
                    else:
                        self.train_imgs['label'].append(torch.tensor(1))
                for path in valid_maj_sample+valid_min_sample:
                    self.valid_imgs['data'].append(str(path))
                    if int(path.parent.name)==args.maj:
                        self.valid_imgs['label'].append(torch.tensor(0))
                    else:
                        self.valid_imgs['label'].append(torch.tensor(1))

            elif dir.name=='test':
                maj_list = list(dir.glob(str(self.args.maj)+'/*.png'))
                min_list = list(dir.glob(str(self.args.min)+'/*.png'))
                min_list = random.sample(min_list,200)
                for path in maj_list+min_list:
                    self.test_imgs['data'].append(str(path))
                    if int(path.parent.name)==args.maj:
                        self.test_imgs['label'].append(torch.tensor(0))
                    else:
                        self.test_imgs['label'].append(torch.tensor(1))
                
        if self.mode=='rb':
            """balanced dataset"""
            min_path = np.array(self.train_imgs['data'])[np.array(self.train_imgs['label'])==1]
            for path in min_path:
                self.rb_imgs['data'].append(str(path))
                self.rb_imgs['label'].append(torch.tensor(1)) 
            "ngにデータ数を合わせる"
            maj_path=np.array(self.train_imgs['data'])[np.array(self.train_imgs['label'])==0]
            maj_path = random.sample(maj_path.tolist(),add_center+len(min_path))
            for path in maj_path:
                self.rb_imgs['data'].append(path)
                self.rb_imgs['label'].append(torch.tensor(0))
        
        # seed値を戻す
        random.seed(self.args.seed)

        if self.mode=='cb':
            for path in paths:
                self.cb_imgs['data'].append(path)
                path = Path(path)
                if path.parent.name == str(self.args.maj):
                    self.cb_imgs['label'].append(torch.tensor(0))
                elif path.parent.name == str(self.args.min):
                    self.cb_imgs['label'].append(torch.tensor(1))
                else:
                    raise ValueError('invalid name file may exist')
        

    def get_num(self,name):
        if name=='train':
            self.train_imgs['label'] = np.array(self.train_imgs['label'])
            return (len(self.train_imgs['label'][self.train_imgs['label']==self.args.maj]),len(self.train_imgs['label'][self.train_imgs['label']==1]))
        elif name=='valid':
            self.valid_imgs['label'] = np.array(self.valid_imgs['label'])
            return (len(self.valid_imgs['label'][self.valid_imgs['label']==self.args.maj]),len(self.valid_imgs['label'][self.valid_imgs['label']==1]))
        elif name=='test':
            self.test_imgs['label'] = np.array(self.test_imgs['label'])
            return (len(self.test_imgs['label'][self.test_imgs['label']==self.args.maj]),len(self.test_imgs['label'][self.test_imgs['label']==1]))
        else:
            raise ValueError('Invalid name')

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



class cifar_loader():
    def __init__(self,args):
        self.args=args
        self.transform = transforms.Compose([
                transforms.ToTensor(),       
                transforms.RandomRotation(90),         
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)), ])     

        self.transform_test = transforms.Compose([
                transforms.ToTensor(),               
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),  ])
            
        self.train_dataset=cifar_dataset(self.args,'train',self.transform,num_samples=50000)
        self.valid_dataset=cifar_dataset(self.args,'valid',self.transform,num_samples=50000)
        self.test_dataset=cifar_dataset(self.args,'test',self.transform,num_samples=50000)

    
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
            rb_dataset = cifar_dataset(self.args,'rb',self.transform,num_samples=50000,paths=path,add_center=add_center)
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
            cb_dataset = cifar_dataset(self.args,'cb',self.transform,num_samples=50000,paths=path)
            cb_loader = DataLoader(
                cb_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=32,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return cb_loader
    def get_num(self,name=None,loader=None):
        if name==None and loader!=None:
            cls_1,cls_2=0,0
            for index,(data,label,_) in enumerate(loader):
                data=data.to(device)
                label=label.to(device)
                cls_1+=len(label[label==0])
                cls_2+=len(label[label==1])
            return [cls_1,cls_2]

        elif name!=None and loader==None:
            if name=='train':
                return self.train_dataset.get_num(name)
            elif name=='valid':
                return self.valid_dataset.get_num(name)
            if name=='test':
                return self.test_dataset.get_num(name)
        

       
if __name__ == '__main__':
    import argparse    
    parser = argparse.ArgumentParser(description='ハイパラに関して')
    parser.add_argument('--seed',type=int,default=9999)
    parser.add_argument('--add_center',type=int,default=0)
    parser.add_argument('--mode',type=str,default='nc')
    parser.add_argument('--maj',type=int,default=5)
    parser.add_argument('--min',type=int,default=1)
    parser.add_argument('--min_class_num',type=int,default=60)
    args = parser.parse_args()

    device = torch.device('cuda')
    loader=cifar_loader(args)
    loader.run('rb')
    tr_list=loader.get_num('train')
    va_list=loader.get_num('valid')
    te_list=loader.get_num('test')
    print(f'train:{tr_list}')
    print(f'valid:{va_list}')
    print(f'test:{te_list}')
    import pdb;pdb.set_trace()
   
    