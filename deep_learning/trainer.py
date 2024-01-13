from collections import defaultdict
from models import MultiCalibration
import torch
import numpy as np
from torch.nn import CrossEntropyLoss,Softplus,LeakyReLU
def get_mask(ytrue):
    return ~torch.isnan(ytrue)
def l2loss(ypred,ytrue,mask,rat):
    mse = torch.pow(ypred - ytrue,2)*mask        
    sc2 = torch.pow(ytrue,2)*mask
    return mse.sum()/rat,sc2.sum()/rat

def criterion(ypred,y,probs):
    mask = get_mask(y)
    y[~mask] = 0
    rat = mask.sum()
    mse,sc2 = l2loss(ypred,y,mask,rat)  
    probs,wd = probs
    eps = 1e-5
    entropy = -torch.log(probs + eps)*probs - torch.log(1 - probs + eps)*(1 - probs)
    entropy = entropy.mean(dim = [2,3])
    entropy = (entropy*mask).sum()/rat
    total =   mse + entropy
    metrics = {}
    
    lrgst_prob = torch.sort(probs.detach(),dim = 2,descending = True)[0][:,:,0,:]
    lrgst_prob = (lrgst_prob.mean(dim = 2)*mask).sum()/rat
    
    metrics['total'] = total.item()
    metrics['mse'] = mse.item()
    metrics['sc2'] = sc2.item() 
    metrics['entrp'] = entropy.item()
    metrics['lrgst_prob'] = lrgst_prob.item()
    return total,metrics

def stochastic_criterion(ypred,y,model):
    yp0,yp1 = ypred
    ym = ~torch.isnan(y)
    badflag = torch.sum(ym) == 0
    yms = torch.split(ym,1,dim = 1)
    ym0,ym1 = tuple(ymi.squeeze() for ymi in yms)
    yt0,yt1 = model.classify(y)
    
    yt0 = yt0[ym0].squeeze()
    yt1 = yt1[ym1].squeeze()
    yp0 = yp0[ym0]
    yp1 = yp1[ym1]
    celoss = CrossEntropyLoss()
    yps = torch.concatenate([yp0,yp1],dim = 0)
    yts = torch.concatenate([yt0,yt1],dim = 0)
    err = celoss(yps,yts) 
    metrics = dict(cross_entropy_loss = err.item())
    return err,badflag,metrics

class Trainer:
    def __init__(self,model:MultiCalibration,optimizer,lrscheduler,data_loader,tag,writer,nfeats_scheduler):
        self.optimizer = optimizer
        self.lrscheduler = lrscheduler
        self.model = model
        self.data_loader = data_loader
        self.tag = tag
        self.writer = writer
        self.counter = 0
        self.nfeats_scheduler = nfeats_scheduler
        self.epoch = 0
    def run_epoch(self,):
        metrics = defaultdict(lambda:0)
        train_flag = 'train' in self.tag
        if train_flag:
            self.model.train()
        else:
            self.model.eval()
        nupdates = 0
        for xt,t,y in self.data_loader:
            if train_flag:
                self.optimizer.zero_grad()
                ypred,probs = self.model(xt,t)
            else:
                with torch.no_grad():
                    ypred,probs = self.model(xt,t)
            total,metric_update = criterion(ypred,y,probs)
            nupdates+=1        
            self.counter += 1
            
            if train_flag:
                total.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.1)
                self.optimizer.step()
            metric_updates = tuple(metric_update.items())
            for key,val in metric_updates:
                metrics[key] += val
                if key == 'sc2':
                    val = 1 - metrics['mse']/metrics[key]
                    key = 'r2'
                else:
                    val = metrics[key]/nupdates
                if train_flag:
                    self.writer.add_scalar(f"per-update-{self.tag}/{key}", val, self.counter)
                    if val > 0:
                        self.writer.add_scalar(f"per-update-{self.tag}/{key}-log-scale", np.log10(val), self.counter)        
        metrics['r2'] = 1 - metrics['mse']/metrics['sc2']
        metric_updates = list(metrics.items())
        for key,val in metric_updates:
            if key != 'r2':
                val = val/nupdates
            metrics[key] = val
            self.writer.add_scalar(f"{self.tag}/{key}", val, self.epoch)
            if val > 0:
                self.writer.add_scalar(f"{self.tag}/{key}-log-scale", np.log10(val), self.epoch)
        self.epoch+=1
        return metrics

