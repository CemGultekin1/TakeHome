from deep_learning.dataset import PriceTransform,CustomDataset
from torch.utils.data import DataLoader
import os 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from models import MultiCalibration,StochasticModel
import torch
from trainer import Trainer
import matplotlib.pyplot as plt

# def compute_normalization_consts():
    

def main():
    ncpu = min(0,os.cpu_count())
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    
    t = 10
    parquet_partition = dict(
        train = parquet_files[t:t+1],
    )
    
    ntime_daily = 4
    num_feats =  60 

    transform = PriceTransform(ntime_daily=ntime_daily,return_order = 'x,mt,y')
    transform.normalization_consts

    def get_loaders(data_partition = 'train'):    
        cdata = CustomDataset(parquet_partition[data_partition],ncpu,transform = transform,internal_shuffle_flag=False,per_request=128)
        return DataLoader(cdata, batch_size=1, shuffle=False, num_workers=ncpu,timeout = 30 if ncpu>0 else 0)

    test_dataloader = get_loaders('train')
    
    model = StochasticModel(*transform.output_dimensions(),num_feats = num_feats,out_dim = 32)
    model_path = 'model_20240104-dummy-0.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_dataloader_iter = iter(test_dataloader)
    
    for _ in range(4):
        x,m,y = next(test_dataloader_iter)
    
    with torch.no_grad():
        yp0,yp1 = model(x.to(torch.float32).squeeze(),m.to(torch.float32).squeeze())
    
    # yp0 = model.net1.adaptive_histogram.pick_medians(yp0)
    # yp1 = model.net2.adaptive_histogram.pick_medians(yp1)
    # 
    # yp0,st0 = model.net1.adaptive_histogram.conditional_mean_std(yp0)
    # yp1,st1 = model.net2.adaptive_histogram.conditional_mean_std(yp1)
    # ppred = torch.stack([st0,st1],dim = 1)
    # ypred = torch.stack([yp0,yp1],dim = 1)
    cp0 = model.net1.adaptive_histogram.get_probs(yp0)
    cp1 = model.net2.adaptive_histogram.get_probs(yp1)
    
    ypred = torch.stack([cp0,cp1],dim = 2)
    def to_numpy(x:torch.Tensor):
        x = x.numpy().astype(float)
        return x
    
    ypred = to_numpy(ypred)   
    print(ypred.shape)
    plt.imshow(ypred[...,0].T)
    plt.savefig('imtest.png')
    return
    # ppred = to_numpy(ppred)   
    y = to_numpy(y).astype(float).squeeze()
    fig,axs = plt.subplots(2,1,figsize = (15,30))
    for i in range(2):#y.shape[1]):
        axs[i].plot(y[:,i],label = f'true-{i}')
        axs[i].plot(ypred[:,i],label = f'pred-{i}',alpha = 0.5)
        
        # axs[i].plot(ypred[:,i] + ppred[:,i],alpha = 0.25,linestyle = '--',color = 'g')
        # axs[i].plot(ypred[:,i] - ppred[:,i],alpha = 0.25,linestyle = '--',color = 'g')
        axs[i].legend()
    fig.savefig('pred.png')
if __name__ == '__main__':
    main()