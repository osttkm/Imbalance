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



class gaps_dataset(Dataset): 
    """シード値の固定"""
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self, root, mode,transform,num_samples=0,paths=[]): 
        self.root=root
        self.mode=mode
        self.transform=transform

        self.train_imgs={
            "data":[],
            "label":[],
        }
        self.test_imgs={
            "data":[],
            "label":[],
        }
        self.sampled_imgs={
            "data":[],
            "label":[],
        }

        data_root = Path(root)
        train_path = data_root / "train"
        test_path = data_root / "test"
        
        train_dir = [path for path in train_path.iterdir() if path.is_dir()]
        test_dir = [path for path in test_path.iterdir() if path.is_dir()]

        for category_dir in train_dir:
            label = category_dir.name
            img_paths = list(category_dir.glob("*.png"))
            img_num = len(img_paths)
            for path in img_paths:
                self.train_imgs['data'].append(str(path))
            if label == "Intact_road":
                for i in range(img_num):
                    self.train_imgs['label'].append(torch.tensor(0))
            else:
                for i in range(img_num):
                    self.train_imgs['label'].append(torch.tensor(1)) 

        for category_dir in test_dir:
            label = category_dir.name
            img_paths = list(category_dir.glob("*.png"))
            img_num = len(img_paths)
            for path in img_paths:
                self.test_imgs['data'].append(str(path))
            if label == "Intact_road":
                for i in range(img_num):
                    self.test_imgs['label'].append(torch.tensor(0))
            else:
                for i in range(img_num):
                    self.test_imgs['label'].append(torch.tensor(1))
        
        for path in paths:
            path = Path(path)
            label = path.parent.name
            img_num = len(paths)
            self.sampled_imgs['data'].append(str(path))
        if label == "Intact_road":
            for i in range(img_num):
                self.sampled_imgs['label'].append(torch.tensor(0))
        else:
            for i in range(img_num):
                self.sampled_imgs['label'].append(torch.tensor(1))

    def __getitem__(self, index):
        if self.mode=='train':
            img = Image.open(self.train_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.train_imgs["label"][index],self.train_imgs["data"][index]
        elif self.mode=='sampled':
            img = Image.open(self.sampled_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.sampled_imgs["label"][index],self.sampled_imgs["data"][index]
        else:
            img = Image.open(self.test_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.test_imgs["label"][index],self.test_imgs["data"][index]
            
        
    def __len__(self):
        if self.mode=="train":
            return len(self.train_imgs["data"])
        if self.mode=="sampled":
            return len(self.sampled_imgs["data"])
        else:
            return len(self.test_imgs["data"])



class gaps_loader():
    def __init__(self,root,batch_size):
        self.batch_size = batch_size
        self.root = root

        """シード値の固定"""
        random_seed = 9999
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        self.transform = transforms.Compose([
                transforms.ToTensor(),                
                transforms.Resize((64,64)),
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)), ])     

        self.transform_test = transforms.Compose([
                transforms.ToTensor(),            
                transforms.Resize((64,64)),    
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),  ])
            
        self.train_dataset=gaps_dataset(self.root,'train',self.transform,num_samples=50000)
        self.test_dataset=gaps_dataset(self.root,'test',self.transform,num_samples=50000)

    
    def run(self,mode,path=[]):
        "trainデータは通常通り8:1:1"
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        ratio = 0.2                              
        valid_size = int(len(self.train_dataset) * ratio)    
        train_size = len(train_dataset) - valid_size 
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    
        if mode=='train':
            train_loader = DataLoader(
                train_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return train_loader
        elif mode=='valid':
            valid_loader = DataLoader(
                valid_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return valid_loader
        elif mode=='test':
            test_loader = DataLoader(
                test_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return test_loader
        elif mode=='sampled':
            sampled_dataset = gaps_dataset(self.root,'sampled',self.transform,num_samples=50000,paths=path)
            sampled_loader = DataLoader(
                sampled_dataset,
                pin_memory=True,
                drop_last=True,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count()
            )
            return sampled_loader
        

       
if __name__ == '__main__':
    loader=gaps_loader('/home/dataset/gaps',256)
    train_data = loader.run('train')
    index = np.arange(10)
    path = []
    for i in index:
        path.append(train_data.dataset[i][2])
    sampled = loader.run('sampled',path)
    import pdb;pdb.set_trace()


          
            

