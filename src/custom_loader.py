from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import numpy as np
from PIL import Image
import torch
import random
import os
from pathlib import Path
import argparse


# parser = argparse.ArgumentParser(description='ハイパラ')
# parser.add_argument('--aug',type=int,default=0)
# parser.add_argument('--seed',type=int,default=9999)
# args = parser.parse_args()


class custom_dataset(Dataset):
    def __init__(self,root,mode,transform,num_samples=0):
        self.root = root
        self.mode = mode
        self.transform=transform
       
        self.train_imgs={
            "data":[],
            "label":[]
        }
        self.valid_imgs={
            "data":[],
            "label":[]
        }
        self.test_imgs={
            "data":[],
            "label":[]
        }
        self.org_train_imgs={
            "data":[],
            "label":[]
        }
        self.random_train_imgs={
            "data":[],
            "label":[]
        }

        for data in glob.glob(self.root+'/train/*/*'):
            label = Path(data).parent.name
            self.train_imgs["data"].append(data)
            self.train_imgs["label"].append(torch.tensor(int(label)))


        for data in glob.glob(self.root+'/valid/*/*'):
            label = Path(data).parent.name
            self.valid_imgs["data"].append(data)
            self.valid_imgs["label"].append(torch.tensor(int(label)))

        for data in glob.glob(self.root+'/test/*/*'):
            label = Path(data).parent.name
            self.test_imgs["data"].append(data)
            self.test_imgs["label"].append(torch.tensor(int(label)))
       
        for data in glob.glob(self.root+'/org_train/*/*'):
            label = Path(data).parent.name
            self.org_train_imgs["data"].append(data)
            self.org_train_imgs["label"].append(torch.tensor(int(label)))

        for data in glob.glob(self.root+'/random_select_train/*/*'):
            label = Path(data).parent.name
            self.random_train_imgs["data"].append(data)
            self.random_train_imgs["label"].append(torch.tensor(int(label)))
            
        
    def __getitem__(self, index):
        if self.mode=='train':
            img = Image.open(self.train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.train_imgs["label"][index]
        elif self.mode=='valid':
            img = Image.open(self.valid_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.valid_imgs["label"][index]
        elif self.mode=='test':
            img = Image.open(self.test_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.test_imgs["label"][index]
        elif self.mode=='random':
            img = Image.open(self.random_train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.random_train_imgs["label"][index]
        else:
            img = Image.open(self.org_train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.org_train_imgs["label"][index]
        
    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs["data"])
        elif self.mode=="valid":
            return len(self.valid_imgs["data"])
        elif self.mode=="test":
            return len(self.test_imgs["data"])
        elif self.mode=="random":
            return len(self.random_train_imgs["data"])
        else:
            return len(self.org_train_imgs["data"])
    
class custom_loader():
    def __init__(self,root,batch_size,args):
        self.batch_size = batch_size
        self.root = root
        self.args = args

        """シード値の固定"""
        random_seed = args.seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        if args.aug == 0:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),                
                    transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),      
                ]) 
        elif args.aug == 1: 
            self.transform = transforms.Compose([
                    transforms.ToTensor(),                
                    transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),      
                    transforms.RandomErasing(p=0.5,scale=(0.02, 0.33)),
                    transforms.RandomRotation(30),
                ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),                
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),      
            ])
    

    def run(self,mode):
        if mode=='train':
            train_dataset=custom_dataset(self.root,mode,self.transform,num_samples=50000)
            train_loader=DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=4)
            return train_loader
        elif mode=='valid':
            valid_dataset=custom_dataset(self.root,mode,self.transform_test,num_samples=50000)
            valid_loader=DataLoader(
                dataset=valid_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=os.cpu_count())
            return valid_loader
        elif mode=='test':
            test_dataset=custom_dataset(self.root,mode,self.transform_test,num_samples=50000)
            test_loader=DataLoader(
                    dataset=test_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=os.cpu_count())
            return test_loader
        elif mode=='random':
            random_train_dataset=custom_dataset(self.root,mode,self.transform,num_samples=50000)
            random_train_loader=DataLoader(
                    dataset=random_train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=os.cpu_count())
            return random_train_loader
        elif mode=='org_train':
            org_train_dataset=custom_dataset(self.root,mode,self.transform,num_samples=50000)
            org_train_loader=DataLoader(
                dataset=org_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=os.cpu_count())
            return org_train_loader

# if __name__=='__main__':
#     loader = custom_loader('./clst20_handle_ng0.5_del300_data',256,args)
#     train_loader = loader.run('train')
#     org_train_loader = loader.run('org_train')
 