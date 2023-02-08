import torch
import numpy as np
import random


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def furthest_first(unlabeled_embeddings, labeled_embeddings, badget, init_num):
    
    """### シード値の固定"""
    random_seed = 43
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    
    unlabeled_embeddings = unlabeled_embeddings.to(device)
    if labeled_embeddings == True:
        labeled_embeddings = labeled_embeddings.to(device)
    m = unlabeled_embeddings.shape[0]
    unlabeled_embeddings = unlabeled_embeddings.reshape(m,-1)
    memory_unlabeled_embeddings = unlabeled_embeddings.reshape(m,-1)
    
    if labeled_embeddings == False:
        min_dist = torch.tile(torch.tensor(float('inf')), (m,1)).flatten().to(device)
    else:
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0].to(device)
    org_var = torch.var(unlabeled_embeddings)
    best_idxs = torch.tensor([]).to(device)
    iner_dis=0
    for i in range(init_num):
        idxs = []
        print('この実験でのみ初期化しているため，本実験以外ではコードの修正が必要です')
        # unlabeled_embeddings = memory_unlabeled_embeddings
        min_dist = torch.tile(torch.tensor(float('inf')), (m,1)).flatten().to(device)
        for k in range(badget):
            # 初期点をランダムに生成
            if k==0:
                idx = torch.randint(low=0,high=len(min_dist),size=(1,)).to(device)
            else: 
                idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:],p=2).flatten() # [data_num,1]
            min_dist = torch.minimum(min_dist, dist_new_ctr)
        iner_dis=torch.var(unlabeled_embeddings[idxs])
        if i==0:
            best_idxs = idxs
        if abs(org_var-torch.var(unlabeled_embeddings[best_idxs])) > abs(org_var-iner_dis) and len(best_idxs)==len(np.unique(best_idxs)):
            best_idxs = idxs
            print(org_var-iner_dis)
    return best_idxs

    


