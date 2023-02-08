from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import glob
import os
import random



class handle_dataset(Dataset): 
    """シード値の固定"""
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self, root, mode,transform,num_samples=0): 
        self.root=root
        self.mode=mode
        self.transform=transform

        self.ok_imgs={
            "data":[],
            "label":[],
            "path":[],
            "t_label":[]
        }
        self.ng_imgs={
            "data":[],
            "label":[],
            "path":[],
            "t_label":[]
        }
      
        for data in glob.glob(self.root+'/OK/8548_01_OK/*.png'):
            self.ok_imgs["data"].append(data)
            self.ok_imgs["label"].append(torch.tensor(0))
            self.ok_imgs["path"].append(data)
            self.ok_imgs['t_label'].append(torch.tensor(0))
        for data in glob.glob(self.root+'/OK/8549_02_OK_window/*.png'):
            self.ok_imgs["data"].append(data)
            self.ok_imgs["label"].append(torch.tensor(0))
            self.ok_imgs["path"].append(data)
            self.ok_imgs['t_label'].append(torch.tensor(1))

        for data in glob.glob(self.root+'/NG/8550_03_frash/*.png'):
            self.ng_imgs["data"].append(data)
            self.ng_imgs["label"].append(torch.tensor(1))
            self.ng_imgs["path"].append(data)
            self.ng_imgs['t_label'].append(torch.tensor(2))
       
        for data in glob.glob(self.root+'/NG/8551_04_flash_window/*.png'):
            self.ng_imgs["data"].append(data)
            self.ng_imgs["label"].append(torch.tensor(1))
            self.ng_imgs["path"].append(data)
            self.ng_imgs['t_label'].append(torch.tensor(3))

        for data in glob.glob(self.root+'/NG/8552_05_fmatter/*.png'):
            self.ng_imgs["data"].append(data)
            self.ng_imgs["label"].append(torch.tensor(1))
            self.ng_imgs["path"].append(data)
            self.ng_imgs['t_label'].append(torch.tensor(4))

        for data in glob.glob(self.root+'/NG/8553_06_white_fmatter/*.png'):
            self.ng_imgs["data"].append(data)
            self.ng_imgs["label"].append(torch.tensor(1))
            self.ng_imgs["path"].append(data)
            self.ng_imgs['t_label'].append(torch.tensor(5))

    def __getitem__(self, index):
        if self.mode=='ok':
            img = Image.open(self.ok_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.ok_imgs["label"][index]
            # ,self.ok_imgs["path"][index],self.ok_imgs["t_label"][index]
        else:
            img = Image.open(self.ng_imgs["data"][index]).convert('RGB')
            img = self.transform(img)
            return img,self.ng_imgs["label"][index]
            # ,self.ng_imgs["path"][index],self.ng_imgs["t_label"][index]
        
        
    def __len__(self):
        if self.mode=="ok":
            return len(self.ok_imgs["data"])
        else:
            return len(self.ng_imgs["data"])



class handle_loader():
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
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)), ])     

        self.transform_test = transforms.Compose([
                transforms.ToTensor(),                
                transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2)),  ])
            
        self.ok_dataset=handle_dataset(self.root,'ok',self.transform,num_samples=50000)
        self.ng_dataset=handle_dataset(self.root,'ng',self.transform,num_samples=50000)
        "ng品の分布について4;1;5にしてみる"
        ratio = 0.4
        ng_size = int(len(self.ng_dataset)*ratio)
        _size = len(self.ng_dataset)-ng_size
        self.ng_valid_dataset, self.ng_dataset = torch.utils.data.random_split(self.ng_dataset, [_size,ng_size])
        ratio = 0.83
        ng_size = int(len(self.ng_valid_dataset)*ratio)
        _size = len(self.ng_valid_dataset)-ng_size
        self.ng_valid_dataset, self.ng_test_dataset = torch.utils.data.random_split(self.ng_valid_dataset, [_size,ng_size])

    
    def run(self,mode):
        # "trainのngデータを調節する"
        # ratio = 0.5
        # ng_size = int(len(self.ng_dataset)*ratio)
        # _size = len(self.ng_dataset)-ng_size
        # _, ng_train_dataset = torch.utils.data.random_split(self.ng_dataset, [_size,ng_size])
    
        "okデータは通常通り8:1:1"
        dataset = self.ok_dataset
        ratio = 0.2                              
        ok_valid_size = int(len(self.ok_dataset) * ratio)    
        ok_train_size = len(dataset) - ok_valid_size 
        ok_train_dataset, ok_valid_dataset = torch.utils.data.random_split(dataset, [ok_train_size, ok_valid_size])
        train_dataset=ok_train_dataset + self.ng_dataset

        ratio = 0.5
        ok_test_size = int(len(ok_valid_dataset) * ratio)    
        ok_valid_size = len(ok_valid_dataset) - ok_test_size
        ok_valid_dataset, ok_test_dataset = torch.utils.data.random_split(ok_valid_dataset, [ok_valid_size, ok_test_size])
        valid_dataset = ok_valid_dataset+self.ng_valid_dataset
        test_dataset = ok_test_dataset+self.ng_test_dataset
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
        

       

if __name__ == '__main__':
    loader=handle_loader('/home/oshita/cleansing/data/handle/gray',256)
    ok_data = loader.run('train')
    import pdb;pdb.set_trace()

          
            

