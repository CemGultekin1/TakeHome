from collections import defaultdict
from deep_learning.pack import ModelPack
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
    probs = probs
    eps = 1e-5
    entropy = -torch.log(probs + eps)*probs - torch.log(1 - probs + eps)*(1 - probs)
    entropy = entropy.mean(dim = [2,3])
    entropy = (entropy*mask).sum()/rat
    total =   mse + entropy*0.01
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
    def __init__(self,models:ModelPack,data_loader,tag):
        self.models = models
        self.data_loader = data_loader
        self.tag = tag
        self.counter = 0
        self.epoch = 0
    def run_epoch(self,):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        [mdl.model.to(device) for mdl in self.models]
        train_flag = 'train' in self.tag
        if train_flag:
            [mdl.model.train() for mdl in self.models]
        else:
            [mdl.model.eval() for mdl in self.models]
        nupdates = 0
        for args in self.data_loader:
            xt,t,y = (g.to(device) for g in args)
            for i,model_optim in enumerate(self.models):
                optimizer = model_optim.optimizer
                model = model_optim.model
                writer = model_optim.writer
                metrics = model_optim.metrics
                if train_flag:
                    optimizer.zero_grad()
                    ypred,probs = model(xt,t)
                else:
                    with torch.no_grad():
                        ypred,probs = model(xt,t)
                total,metric_update = criterion(ypred,y,probs)
                if i== 0:
                    nupdates+=1        
                    self.counter += 1
                
                if train_flag:
                    total.backward()
                    optimizer.step()
                metric_updates = tuple(metric_update.items())
                for key,val in metric_updates:
                    metrics[key] += val
                    if key == 'sc2':
                        val = 1 - metrics['mse']/metrics[key]
                        key = 'r2'
                    else:
                        val = metrics[key]/nupdates
                    if train_flag:
                        writer.add_scalar(f"per-update-{self.tag}/{key}", val, self.counter)
                        if val > 0:
                            writer.add_scalar(f"per-update-{self.tag}/{key}-log-scale", np.log10(val), self.counter)        
        for mdl_opt in self.models:
            metrics = mdl_opt.metrics
            writer = mdl_opt.writer
            metrics['r2'] = 1 - metrics['mse']/(metrics['sc2']+1e-19)
            metric_updates = list(metrics.items())
            for key,val in metric_updates:
                if key != 'r2':
                    val = val/nupdates
                metrics[key] = val
                writer.add_scalar(f"{self.tag}/{key}", val, self.epoch)
                if val > 0:
                    writer.add_scalar(f"{self.tag}/{key}-log-scale", np.log10(val), self.epoch)
        self.epoch+=1

