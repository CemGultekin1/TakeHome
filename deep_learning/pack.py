from typing import List
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from models import MultiCalibration
import torch
from collections import defaultdict


class EntropyControl:
    def __init__(self,maxcoeff) -> None:
        self.maxcoeff = maxcoeff
        self.coeff = 1.
    def step(self,lrgst_prob):
        if lrgst_prob >0.95:
            self.coeff = self.maxcoeff
        self.coeff *= 1.5
        self.coeff = np.minimum(self.coeff,self.maxcoeff)
    def is_saturated(self,):
        return self.coeff == self.maxcoeff
    def __call__(self,):
        return self.coeff





class ModelOptimizerPair:
    def __init__(self,nfeats):
        self.entropy_control = EntropyControl(256)
        x_dim = 375
        ntime_daily = 4
        self.nfeats = nfeats
        self.model = MultiCalibration(x_dim,t_dim = ntime_daily,n_feats = nfeats,lyr_widths = [16]*4,stress_control=self.entropy_control)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= 1e-3)        
        self.scheduler = ReduceLROnPlateau(self.optimizer,mode = 'max',min_lr = 1e-7,patience = 5,factor = 0.1)
        self.model_path = f'model_{nfeats}_epoch.pth'
        self.writer = SummaryWriter(comment = self.model_path.replace('.pth',''))
        self.metrics = defaultdict(lambda :0)
    def end_train_epoch(self,epoch:int):
        model_path = self.model_path.replace('epoch',str(epoch))
        torch.save(self.model.state_dict(), model_path.replace('.pth',f'-{epoch}.pth'))
        self.entropy_control.step(self.metrics['lrgst_prob'])
        self.writer.flush()
        self.metrics = defaultdict(lambda :0)
    def begin_epoch(self,epoch):
        print(f' nfeats = {self.nfeats} started epoch = {epoch}')
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar(f"log-lr",np.log10(lr),epoch)
        self.metrics = defaultdict(lambda :0)
    def end_validation_epoch(self,):
        if self.entropy_control.is_saturated():
            self.scheduler.step(self.metrics['r2'])
        self.writer.flush()
                
class ModelPack(list):
    def __init__(self,nfeats:List[int]) -> None:
        super().__init__()
        for nf in nfeats:
            self.append(ModelOptimizerPair(nf))