from typing import List
from deep_learning.dataset import PriceTransform,CustomDataset
from torch.utils.data import DataLoader
import os 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from models import MultiCalibration
import torch
from trainer import Trainer


class EntropyControl:
    def __init__(self,maxcoeff) -> None:
        self.maxcoeff = maxcoeff
        self.coeff = 1.
    def step(self,lrgst_prob):
        if lrgst_prob < 0.9:
            self.coeff *= 2.
        self.coeff = np.minimum(self.coeff,self.maxcoeff)
    def is_saturated(self,):
        return self.coeff == self.maxcoeff
    def __call__(self,):
        return self.coeff

def custom_collate(batch):
    outs = tuple(zip(*batch))
    return tuple(torch.from_numpy(np.concatenate(out,axis = 0)).type(torch.float32) for out in outs)
    
def main():
    ncpu = min(4,os.cpu_count())
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    n = len(parquet_files)
    # np.random.shuffle(parquet_files)
    per_cpu_files = 100
    parquet_partition = dict(
        train = parquet_files[max((n - ncpu*per_cpu_files),100):],
        val = parquet_files[:min(ncpu*per_cpu_files,100)]
    )

    ntime_daily = 4
    
    transform = PriceTransform(ntime_daily=ntime_daily,return_order = 'xt,t,y')
    transform.normalization_consts
    def get_loaders(data_partition = 'train',ncpu = 0,batch_size = 128/8,per_request = 8): 
        np.random.shuffle(parquet_partition[data_partition])
        cdata = CustomDataset(parquet_partition[data_partition],ncpu,\
            transform = transform,per_request= per_request)        
        return DataLoader(cdata, batch_size=batch_size, shuffle=False,\
                    num_workers=ncpu,timeout = 0 if ncpu == 0 else 120,\
                        collate_fn= custom_collate)

    batch_size = max(ncpu,1)
    per_request = 128
    train_dataloader = get_loaders('train',ncpu = ncpu,batch_size=batch_size,per_request=per_request)
    val_dataloader = get_loaders('val',ncpu = ncpu,batch_size=batch_size,per_request=per_request)
    width = 16
    x_dim = 375+ntime_daily
    nfs = EntropyControl(256)
    
    model = MultiCalibration(x_dim,t_dim = ntime_daily,n_feats = width,lyr_widths = [width]*4,stress_control=nfs)
    # model.load_state_dict(torch.load('model_20240104-dummy-5.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    writer = SummaryWriter()
    scheduler = ReduceLROnPlateau(optimizer,mode = 'max',min_lr = 1e-7,patience = 5,factor = 0.1)

    
    trn = Trainer(model,optimizer,scheduler,train_dataloader,'train',writer,nfs)
    val = Trainer(model,optimizer,scheduler,val_dataloader,'val',writer,nfs)

    model_path = 'model_20240104-dummy.pth'
    for epoch in range(100):    
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f"log-lr",np.log10(lr),epoch)
        train_loss = trn.run_epoch()
        val_loss = val.run_epoch()
        torch.save(model.state_dict(), model_path.replace('.pth',f'-{epoch}.pth'))
        print(f"epoch = {epoch}\t\t train_loss = {train_loss['r2']},\t val_loss = {val_loss['r2']},lr = {lr}, entrcoef = {nfs()}")
        nfs.step(train_loss['lrgst_prob'])
        # scheduler.step()
        writer.flush()
    writer.close()

if __name__ == '__main__':
    main()