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
        # lyr_widths = [in_dims,32,32,self.out_dims*self.nfeats]     
        self.attention = nn.Linear(in_dims,self.out_dims*self.nfeats)   
        # self.sequential = create_sequential(lyr_widths,p = 0)      
        self.softmax = nn.Softmax(dim = 1)        
        self.stress_control = stress_control
        self._probs = None
    def forward(self,xt,t): 
        # h = self.sequential(t)
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
        return y,torch.square(self.sequential.weight.data).mean()

class MultiCalibration(nn.Module):
    net_class = FCNN
    def __init__(self,x_dim,t_dim = 4,n_feats:int = None,stress_control = None,**kwargs):        
        super().__init__()
        self.fs1 = FeatureSelector(t_dim,x_dim,n_feats,stress_control)
        self.fs2 = FeatureSelector(t_dim,x_dim,n_feats,stress_control)
        x_dim = n_feats
        self.net1 = self.net_class(x_dim,**kwargs)
        self.net2 = self.net_class(x_dim,**kwargs)
    def forward(self,*args):
        x,t = args
        x2,probs2 = self.fs2(x,t)
        x1,probs1 = self.fs1(x,t)
        y1,wd1 = self.net1.forward(x1)
        y2,wd2 = self.net2.forward(x2)
        wd = (wd1 + wd2)/2
        return torch.cat([y1,y2],dim = 1),(torch.stack([probs1,probs2],dim = 1),wd)
    
    
class StochasticFCNN(FCNN):
    def __init__(self, *dims, num_feats=10, out_dim=64, lyr_widths=[-1, 64, 64, 64, -1]):
        super().__init__(*dims, num_feats=num_feats, out_dim = out_dim, lyr_widths = lyr_widths)
        # self.softmax = nn.Softmax(dim = 1)
        self.adaptive_histogram = AdaptiveHistogram(n_bins=out_dim,excess_tail=0.1)
    def classify(self,y):
        return self.adaptive_histogram(y)
    def forward(self, x, nanmask_time_info):
        y = super().forward(x, nanmask_time_info)
        return y#self.softmax(y)
class StochasticModel(MultiCalibration):
    net_class = StochasticFCNN
    def classify(self,ys):
        y0,y1 = torch.split(ys,1,dim = 1)
        i0 = self.net1.classify(y0)
        i1 = self.net2.classify(y1)
        return i0,i1 
    def forward(self,*args):
        y1 = self.net1.forward(*args)
        y2 = self.net2.forward(*args)
        return y1,y2
    
class AdaptiveHistogram(nn.Module):
    def __init__(self,n_bins:int,set_extremes_at:int = 1024,excess_tail:float = 0.1) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.extremums = nn.Parameter(torch.tensor([np.inf,-np.inf,excess_tail]),requires_grad=False)
        self.num_samples = nn.Parameter(torch.zeros(1,dtype = torch.int),requires_grad=False)
        self.set_extremes_at = nn.Parameter(torch.ones(1,dtype = torch.int)*set_extremes_at,requires_grad=False)
    def is_init(self,):
        return self.extremums[0] != np.inf and self.extremums[1] != -np.inf
    def update_extremes(self,x):
        x = x[x==x]
        xmax = torch.amax(x)
        xmin = torch.amin(x)
        self.extremums[0] = min(self.extremums[0],xmin)
        self.extremums[1] = max(self.extremums[1],xmax)
        self.num_samples += len(x[:])
    def get_probs(self,yp):
        sf = nn.Softmax(dim = 1)
        cl = sf(yp)
        return cl
    def pick_medians(self,yp):
        sf = nn.Softmax(dim = 1)
        cl = sf(yp).max(1).indices
        xmin,xmax = self.extended_extremums
        edges = torch.linspace(xmin,xmax,self.n_bins+1)
        mids =( edges[:-1] + edges[1:])/2
        print(cl.shape,mids.shape)
        y = mids[cl]
        return y
    def conditional_mean_std(self,yp):
        sf = nn.Softmax(dim = 1)
        cl = sf(yp)
        xmin,xmax = self.extended_extremums
        edges = torch.linspace(xmin,xmax,self.n_bins+1).type(torch.float32)
        mids =( edges[:-1] + edges[1:])/2
        
        avg = torch.einsum('ij,j->i',cl,mids)
        sc2 = torch.einsum('ij,j->i',cl,mids**2)
        std = torch.sqrt(sc2 - avg**2.)
        return avg,std
    @property
    def extended_extremums(self,):
        dext = self.extremums[1] - self.extremums[0]
        xmin = self.extremums[0] - dext*self.extremums[2]
        xmax = self.extremums[1] + dext*self.extremums[2]
        return xmin,xmax
    def __call__(self, x):
        
        if self.training and self.num_samples.item() < self.set_extremes_at.item():
            self.update_extremes(x)
        else:
            if not self.is_init():
                raise Exception('Model hasn\'t been initiated')
        m = torch.isnan(x)
        x[m] = 0
        m = m.squeeze()
        xmin,xmax = self.extended_extremums
        inds = (x - xmin)/(xmax - xmin)*self.n_bins
        inds = torch.maximum(torch.zeros(1,),inds)
        inds = torch.minimum(torch.ones(1,)*(self.n_bins - 1),inds)
        inds = torch.floor(inds).type(torch.long)    
        inds[m] = -1
        # clss = torch.zeros(inds.shape[0],self.n_bins,dtype = torch.float32)
        # clss[torch.arange(inds.shape[0]),inds] = 1.        
        # clss[m,:] = np.nan*torch.ones(1,)
        return inds
        
        
        
        
        