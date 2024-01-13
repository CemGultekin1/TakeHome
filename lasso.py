import itertools
import dask.dataframe as dataframe
import os
import numpy as np
from numpy.random import randint,rand
import matplotlib.pyplot as plt
from data.statistics import get_values,get_statistics_path
import dask
from distributed import Client
import tempfile
from sklearn.linear_model import ElasticNet

def impose_pos_def(mat):
    u,s,_ = np.linalg.svd(mat)
    return u@np.diag(s)@u.T



class Dataset:
    def __init__(self,time_partition,train_flag :bool = True):
        path = get_statistics_path(time_partition,train_flag=train_flag)
        x = np.load(path,allow_pickle=True)[()]
        x = {key:val.astype(np.double) for key,val in x.items()}
        self.XX,self.XY,self.YY = x['xx'],x['xy'],x['yy']
        self.XX = impose_pos_def(self.XX)
        
    def solve_system(self,c = None):
        if c is None:
            c = np.ones(self.XX.shape[0],dtype = bool)
        xx = self.XX[c,:]
        xx = xx[:,c]
        xy = self.XY[c]
        try:
            w_ = np.linalg.solve(xx,xy).squeeze()
        except:
            return None
        w = np.zeros(self.XX.shape[0])
        w[c] = w_
        return w
    def get_rsqr_value(self,w):
        err = w.T@(self.XX@w) - 2*(w.T@self.XY) + self.YY
        assert(err > 0)
        r2 = 1 - err/self.YY
        return r2.squeeze()
def train_lasso(trainset:Dataset,testset:Dataset,alpha = 0.1,l1 = 0.5):
    clf =ElasticNet(alpha = alpha,l1_ratio=l1,fit_intercept = False,max_iter = 2000)
    xmean = np.mean(np.diag(trainset.XX))
    clf.fit(trainset.XX/xmean,trainset.XY/xmean)
    w = clf.coef_.T
    r2 = testset.get_rsqr_value(w).item()
    return w,r2
    
def search_lasso(trainset:Dataset,testset:Dataset):
    alphas = np.power(10,np.linspace(-6,-2,8))
    l1rats = np.linspace(0.1,0.9,8)
    results = []
    for alpha,l1rat in itertools.product(alphas,l1rats):
        _,r2 = train_lasso(trainset,testset,alpha= alpha,l1 = l1rat)
        results.append((alpha,l1rat,r2))
    return results
def lasso_search(time_partition = 0):
    trainset = Dataset(0,train_flag=True)
    testset = Dataset(0,train_flag=False)
    reslts = search_lasso(trainset,testset)
    for res in reslts:
        print(res)
def main():    
    lasso_search()

def plot_summary(time_part = 0):
    solsdict =np.load(f'genetic_alg_results_{time_part}.npy',allow_pickle=True)[()]
    sps = []
    ys = []
    for i,(_,sol) in enumerate(solsdict.items()):
        nz = np.sum(np.abs(sol['weights'])>1e-9)
        sp = sol['sparsity_coeff']
        tr2 = sol['train_r2']
        ts2 = sol['test_r2']
        plt.plot(sol['evals'])
        spstr=np.format_float_scientific(sp, unique=False, precision=3)
        
        plt.title(f'sparsity coeff = {spstr}')
        plt.savefig(os.path.join('imgs',f'spc_{i}.png'))
        plt.close()
        
        sps.append(sp)
        ys.append((nz,tr2,ts2))
    nzs,tr2s,ts2s = tuple(zip(*ys))
    fig,axs = plt.subplots(1,3,figsize = (15,5))
    axs[0].semilogx(sps,nzs,'.')
    axs[1].semilogx(sps,ts2s,'.')
    axs[2].semilogx(sps,tr2s,'.')
    fig.savefig(f'summary_{time_part}.png')
    
    
    
def inspect(file_ids,time_part):
    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    dfs = get_dataset()
    solsdict =np.load(f'genetic_alg_results_{time_part}.npy',allow_pickle=True)[()]
    sols = []
    for _,sol in solsdict.items():      
        w =   sol['weights']
        spc = sol["sparsity_coeff"]
        print(f'coeff = {spc}, #nz = {len(w[np.abs(w)>1e-9])}')
        sols.append((spc,w))
    for fid in file_ids:
        x,y = get_values(fid,(time_part,5),dfs)
        
        glbr2 = 1 - np.linalg.norm(x@w - y)**2/np.linalg.norm(y)**2
        glbr2 = np.format_float_scientific(glbr2, unique=False, precision=3)
        
        nt =1000
        tt = np.floor(x.shape[0]/nt).astype(int)
        for i in range(tt):
            tslc = slice(i*nt,(i+1)*nt)
            x1 = x[tslc,:]
            y1 = y[tslc]
            
            for j,(spc,w) in enumerate(sols):
                plt.plot(y1,label = 'ytrue')                           
                title = f'glbl = {glbr2},'
                ypred = x1@w
                lclr2 = 1 - np.linalg.norm(ypred - y1)**2/np.linalg.norm(y1)**2                
                plt.plot(ypred,label = f'ypred-{spc}')
                lclr2 = np.format_float_scientific(lclr2, unique=False, precision=3)
                title += f' {lclr2},'
                plt.legend()
                plt.title(title)
                plt.savefig(f'imgs/pred_{fid}_{i}_{j}.png')
                plt.close()
if __name__ == '__main__':
    main()