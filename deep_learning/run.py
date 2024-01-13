from typing import List
from deep_learning.dataset import PriceTransform,CustomDataset
from deep_learning.pack import ModelPack
from torch.utils.data import DataLoader
import os 
import numpy as np
import torch
from trainer import Trainer



def custom_collate(batch):
    outs = tuple(zip(*batch))
    return tuple(torch.from_numpy(np.concatenate(out,axis = 0)).type(torch.float32) for out in outs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"working with device = {device}")
    ncpu = min(8,os.cpu_count())
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    n = len(parquet_files)
    # np.random.shuffle(parquet_files)
    per_cpu_files = 1
    parquet_partition = dict(
        train = parquet_files[max((n - ncpu*per_cpu_files),100):],
        val = parquet_files[:min(ncpu*per_cpu_files,100)]
    )

    ntime_daily = 4
    
    transform = PriceTransform(ntime_daily=ntime_daily,return_order = 'x,t,y')
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

    models = ModelPack([2,4,8,16,32])
    
    trn = Trainer(models,train_dataloader,'train')
    val = Trainer(models,val_dataloader,'val')

    for epoch in range(100):    
        [mdl.begin_epoch(epoch) for mdl in models]
        trn.run_epoch()
        [mdl.end_train(epoch) for mdl in models]
        val.run_epoch()
        [mdl.end_epoch() for mdl in models]
    [mdl.close() for mdl in models]

if __name__ == '__main__':
    main()