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

    memory_unlabeled_embeddings = unlabeled_embeddings
    unlabeled_embeddings = unlabeled_embeddings.to(device)
    if labeled_embeddings == True:
        labeled_embeddings = labeled_embeddings.to(device)
    m = unlabeled_embeddings.shape[0]
    unlabeled_embeddings = unlabeled_embeddings.reshape(m,-1)
    
    if labeled_embeddings == False:
        min_dist = torch.tile(torch.tensor(float('inf')), (m,1)).flatten().to(device)
    else:
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0].to(device)
    org_var = torch.var(unlabeled_embeddings)
    best_idxs = torch.tensor([]).to(device)
    iner_dis=0
    best_init_num=0
    reset_flag=False
    for i in range(init_num):
        idxs = []
        history=torch.tensor(0)
        for k in range(badget):
            # 初期点をランダムに生成
            if k==0:
                idx = torch.randint(low=0,high=m,size=(1,)).to(device)
            else: 
                idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:],p=2).flatten() # [data_num,1]
            min_dist = torch.minimum(min_dist, dist_new_ctr)
            min_dist[idx] = -torch.tensor(float('inf'))

        iner_dis=torch.var(unlabeled_embeddings[idxs])
        if len(idxs)!=len(np.unique(idxs)):
            print('distance is resetted')
            min_dist = torch.tile(torch.tensor(float('inf')), (m,1)).flatten().to(device)
            history = min_dist
        else:
            history = min_dist
        if i==0:
            best_idxs = idxs
        if (abs(org_var-torch.var(unlabeled_embeddings[best_idxs])) > abs(org_var-iner_dis)) and (len(best_idxs)==len(np.unique(best_idxs))):
            best_idxs = idxs
            print(org_var-iner_dis)
        if i!=0 and (abs(org_var-torch.var(unlabeled_embeddings[best_idxs])) < abs(org_var-iner_dis)):
            print('not better one')
            min_dist = history
          
        # if i!=0 and  (len(idxs)!=len(np.unique(idxs))):
        #     min_dist = history[i-1]
    return best_idxs



if __name__ == '__main__':
    from shoden_loader import shoden_loader
    import argparse
    import torchvision.models as models
    parser = argparse.ArgumentParser(description='ハイパラに関して')
    parser.add_argument('--seed',type=int,default=9999)
    args = parser.parse_args()
    device = torch.device('cuda')
    loader = shoden_loader(args)
    train = loader.run('train',add_center=300)
    model = models.resnet18(pretrained=True)
    model.eval()
    model.to(device)

    '''coreset'''
    feat = torch.tensor([]).to(device)
    label = torch.tensor([]).to(device)
    with torch.no_grad():
        for index,batch in enumerate(train):
            data,_label,path = batch
            data = data.to(device)
            _label = _label.to(device)  
            feature = model.avgpool(model.layer3(model.layer2(model.layer1 \
                (model.relu(model.bn1((model.conv1(data))))))))
            feat = torch.cat((feat,feature),dim=0)
            label = torch.cat((label,_label),dim=0)
    import pdb;pdb.set_trace()
    ok_feature = feat.reshape(19712,256)[label==0]
    org_var = torch.var(ok_feature)
    index = furthest_first(ok_feature,False,300,19712,args.seed)