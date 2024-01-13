from deep_learning.dataset import PriceTransform,CustomDataset
from torch.utils.data import DataLoader
import os 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from models import MultiCalibration
import torch
from trainer import Trainer
import matplotlib.pyplot as plt

# def compute_normalization_consts():
    

def main():
    ncpu = min(0,os.cpu_count())
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    
    t = 130
    parquet_partition = dict(
        train = parquet_files[t:t+1],
    )
    
    ntime_daily = 12
    num_feats =  1000

    transform = PriceTransform(ntime_daily=ntime_daily,return_order =  'xt,t,y')
    transform.normalization_consts

    def get_loaders(data_partition = 'train'):    
        cdata = CustomDataset(parquet_partition[data_partition],ncpu,transform = transform,internal_shuffle_flag=False,per_request=512)
        return DataLoader(cdata, batch_size=1, shuffle=False, num_workers=ncpu,timeout = 30 if ncpu>0 else 0)

    test_dataloader = get_loaders('train')
    
    model = MultiCalibration(*transform.output_dimensions(),\
            num_feats = num_feats,lyr_widths = [-1,64,64,64,64,64,-1])
    # model_path = 'model_20240104-dummy-8.pth'
    # model.load_state_dict(torch.load(model_path))
    
    # test_dataloader_iter = iter(test_dataloader)
    
    # for _ in range(3):
    #     x,m,y = next(test_dataloader_iter)
    model.load_state_dict(torch.load('diagnosis_path.pth'))
    x = np.load('diagnosis_x.npy')
    y = np.load('diagnosis_y.npy')
    i = np.argmax(np.mean(np.abs(x),axis = 1))
    xa,_ = transform.normalization_consts
    plt.plot(x[i,:375]*xa)
    plt.savefig('x_plot.png')
    plt.close()
    plt.plot(y[i])
    plt.savefig('y_plot.png')
    plt.close()
    model.eval()
    return
    with torch.no_grad():
        ypred = model(x.to(torch.float32).squeeze(),m.to(torch.float32).squeeze())
        # probs1 = model.net1.feature_selector.get_probs(time_xmask)
        # probs2 = model.net2.feature_selector.get_probs(time_xmask)
        # x1,_ = model.net1.feature_selector.forward(time_xmask,x)
        # x2,_ = model.net1.feature_selector.forward(time_xmask,x)
        
    def to_numpy(x:torch.Tensor):
        x = x.numpy().astype(float)
        return x
    ypred = to_numpy(ypred)   
    y = to_numpy(y).astype(float).squeeze()
    ypred = np.where(np.isnan(y),np.nan,ypred)
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