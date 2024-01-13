from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
    
class SkipConnLayer(nn.Module):
    def __init__(self, indim,outdim,p = 0.2) -> None:
        super().__init__()
        self.linlyr = nn.Linear(indim,outdim)
        self.nrmlyr = nn.BatchNorm1d(outdim)
        self.rlulyr = nn.ReLU()
        self.identity_dim = indim== outdim
        self.dropout = nn.Dropout1d(p)
    def forward(self,x):        
        y = self.dropout(self.rlulyr(self.nrmlyr(self.linlyr(x))))
        if self.identity_dim:
            y = y+x
        return  y
    
def create_sequential(lyr_widths,p = 0.2):
    layers = []
    for i in range(len(lyr_widths)-1):
        indim = lyr_widths[i]
        outdim = lyr_widths[i+1]            
        layers.append((f'lyr{i}',SkipConnLayer(indim,outdim,p = p)))
    return nn.Sequential(OrderedDict(layers))

class FeatureSelector(nn.Module):
    def __init__(self, in_dims,out_dims,nfeats,stress_control = None) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.nfeats = nfeats
        self.attention = nn.Linear(in_dims,self.out_dims*self.nfeats)   
        self.softmax = nn.Softmax(dim = 1)        
        self.stress_control = stress_control
        self._probs = None
    def forward(self,xt,t): 
        h = self.attention(t)
        probs = h.reshape([-1,self.out_dims,self.nfeats])
        if self.stress_control is None:
            probs = self.softmax(probs)
        else:
            st = self.stress_control()
            probs = self.softmax(probs*st)
        if not self.training or self.stress_control.is_saturated():
            probs = probs == torch.amax(probs,dim = 1,keepdim=True)
            probs = probs.type(torch.float32)
        y = torch.einsum('ijk,ij->ik',probs,xt)/np.sqrt(self.out_dims)
        return y,probs
    
class FCNN(nn.Module):
    def __init__(self,x_dim,out_dim = 1,lyr_widths = [-1,64,64,64,-1]):        
        super().__init__()
        self.x_dim = x_dim
        lyr_widths[0] = x_dim
        lyr_widths[-1] = out_dim
        self.sequential = nn.Linear(x_dim,out_dim)
        # self.sequential = create_sequential(lyr_widths)
    def forward(self,x):  
        y = self.sequential(x)
        return y

class MultiCalibration(nn.Module):
    net_class = FCNN
    def __init__(self,x_dim,t_dim = 4,n_feats:int = None,stress_control = None,**kwargs):        
        super().__init__()
        self.fs1 = FeatureSelector(t_dim,x_dim,n_feats,stress_control)
        self.fs2 = FeatureSelector(t_dim,x_dim,n_feats,stress_control)
        x_dim = n_feats
        self.net1 = self.net_class(x_dim + t_dim,**kwargs)
        self.net2 = self.net_class(x_dim + t_dim,**kwargs)
    def forward(self,*args):
        x,t = args
        x2,probs2 = self.fs2(x,t)
        x1,probs1 = self.fs1(x,t)
                
        x2 = torch.cat([x2,t],dim = 1)
        x1 = torch.cat([x1,t],dim = 1)
                
        y1 = self.net1.forward(x1)
        y2 = self.net2.forward(x2)
        return torch.cat([y1,y2],dim = 1),torch.stack([probs1,probs2],dim = 1)