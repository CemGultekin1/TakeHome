from typing import Tuple
from data.base import BaseData,DataCleaner,read_parquet
import numpy as np
import os
import dask
from distributed import Client

FEATURE_DIR = 'features'
CUR_FEATURE_NAME = 'base_case'
CONSTS = 'consts'
HISTOGRAMS = 'histograms'
FEATURE_FOLDER_POINTER = os.path.join(FEATURE_DIR,CUR_FEATURE_NAME)
if not os.path.exists(FEATURE_FOLDER_POINTER):
    os.makedirs(FEATURE_FOLDER_POINTER)
class FeatureInfo(BaseData):
    train_parquet_files :Tuple[int,int] = (100,298)
    save_dir:str = os.path.join(FEATURE_DIR)
    def __init__(self,) -> None:
        super().__init__()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

class Categoricals(FeatureInfo):
    file_name :str = 'histograms'
    def __init__(self,nbins:int = 2**10+1,range:Tuple[int,int] = (-32.,32)) -> None:
        super().__init__()
        self.nbins = nbins
        self.range = range
    @property
    def edges(self,):
        return np.linspace(self.range[0],self.range[1],self.nbins+1)
    def collect_histograms(self,):
        @dask.delayed
        def loop_operation(i,x_std,path_cnts,path_ctgr):
            dfs = read_parquet()
            df = dfs.get_partition(i)
            cdt = DataCleaner(allow_bad_x_density=0.,allow_bad_y_density=0.,fill_value=np.nan,remove_rows=True)
            x,_ =cdt(df)
            assert(not np.any(np.isnan(x)))
            del dfs
            x = x/(x_std + 1e-5)
            cnts = []
            ctgr = []
            for i in range(375):
                xi = x[:,i]
                hist,edges = np.histogram(xi,bins = self.nbins,range = self.range)
                cxi = (xi == 0)*0 + (xi >0 )*1 + (xi < 0)*(-1)
                chist,_ = np.histogram(cxi,bins = 3,range=(-1.5,1.5))
                cnts.append(hist)
                ctgr.append(chist)
            np.save(path_cnts,np.stack(cnts,axis = 0))
            np.save(path_ctgr,np.stack(ctgr,axis = 0))
            return
        def get_stds(i):
            dfs = read_parquet()
            df = dfs.get_partition(i)
            cdt = DataCleaner(allow_bad_x_density=0.,allow_bad_y_density=0.,fill_value=np.nan,remove_rows=True)
            x,_ =cdt(df)
            assert(not np.any(np.isnan(x)))            
            del dfs
            x_std = np.std(x,axis = 0)
            return x_std
        x_std = get_stds(self.train_parquet_files[0])
        futures = []
        for i in range(self.train_parquet_files[0],self.train_parquet_files[1]):
            path_cnts = self.path + f'_cnts_{i}'
            path_ctgr = self.path + f'_ctgr_{i}'
            futures.append(loop_operation(i,x_std,path_cnts,path_ctgr))
        Client().compute(futures,sync = True)
        cnts = np.zeros((375,self.nbins),dtype = float)
        ctgr = np.zeros((375,3),dtype = float)
        for i in range(self.train_parquet_files[0],self.train_parquet_files[1]):
            path_cnts = self.path + f'_cnts_{i}.npy'
            path_ctgr = self.path + f'_ctgr_{i}.npy'
            cnts += np.load(path_cnts)
            ctgr += np.load(path_ctgr)
            os.remove(path_ctgr)
            os.remove(path_cnts) 
        cnts = cnts/np.sum(cnts,axis = 1,keepdims=True)       
        ctgr = ctgr/np.sum(ctgr,axis = 1,keepdims=True)       
        save_dict = dict(
            cnts = cnts,
            ctgr = ctgr,
            std = x_std
        )
        for key,val in save_dict.items():
            np.save(self.path + f'_{key}',val)
    @property
    def path(self,):
        return os.path.join(self.save_dir,self.file_name)
    def read_histograms(self,):
        keys = 'cnts ctgr std'.split()
        vals = []
        for key in keys:
            path = np.load(self.path + f'_{key}.npy')
            vals.append(path)
        return vals
    def determine_ctgr(self,**kwargs):
        cnts,ctgr,stds = self.read_histograms()
        zero_concentration = ctgr[:,1] 
        return CategoricalFeatureTransform(cnts,zero_concentration,stds,**kwargs)
    
    
    
class CategoricalFeatureTransform:
    def __init__(self,cnts_hist,zero_concentration,stds,std_upper_limit:float = 30.) -> None:
        ctgr_flag = zero_concentration > 1e-1
        self.cnts_hist = cnts_hist
        self.ctgr_flag = ctgr_flag
        self.stds = stds
        self.std_upper_limit = std_upper_limit
        if np.any(np.isnan(stds)):
                raise Exception(stds)
    @property
    def num_features(self,):
        num_ctgr = np.sum(self.ctgr_flag)
        num_tot = len(self.ctgr_flag)
        return (num_tot - num_ctgr)# + num_ctgr*3
    def __call__(self,x:np.ndarray):
        initially_flat_flag=  x.ndim == 1
        if initially_flat_flag:
            x = x.reshape([1,-1])
        cntsx = x[:,~self.ctgr_flag]   
        if initially_flat_flag:
            return cntsx.flatten()
        return cntsx
        


class LargeFeatures:
    def __init__(self,cnts_hist,zero_concentration,stds,std_upper_limit:float = 30.) -> None:
        ctgr_flag = zero_concentration >1e-3
        self.cnts_hist = cnts_hist
        self.ctgr_flag = ctgr_flag
        self.stds = stds
        self.std_upper_limit = std_upper_limit
        if np.any(np.isnan(stds)):
                raise Exception(stds)
        cumu = np.cumsum(cnts_hist, axis = 1)
        break_pts = []
        p = 1
        P = 1
        for k in range(3):
            p /= 2
            P *= 2
            for i in range(1,P):
                if i%2 == 0:
                    continue
                bp = np.zeros(cumu.shape[0])
                for j in range(cumu.shape[0]):
                    ind = np.where(cumu[j] < i*p)[0][-1]
                    bp[j] = cumu[j][ind]
                if k == 0:
                    break_pts.append((bp,-1))
                    break_pts.append((bp,1))
                else:
                    if i*p < 0.5:
                        break_pts.append((bp,-1))
                    else:
                        break_pts.append((bp,1))
        self.break_pts = break_pts
    def feature_flags(self,):
        flags = [np.zeros(self.num_features,dtype = bool)]*(1+len(self.break_pts))
        flags[:self.ctgr_flag]
    @property
    def num_features(self,):
        num_ctgr = np.sum(self.ctgr_flag)
        num_cnts = len(self.ctgr_flag)*len(self.break_pts)     
        return num_cnts + num_ctgr + 1
    def __call__(self,x:np.ndarray):
        initially_flat_flag=  x.ndim == 1
        if initially_flat_flag:
            x = x.reshape([1,-1])
        ctrx = x[:,self.ctgr_flag]
        cntsx = x#[:,~self.ctgr_flag]
        stds = self.stds.reshape([1,-1]) #[~self.ctgr_flag]
        ctrx = np.concatenate([ctrx == 0],axis = 1).astype(float)
        upl = self.std_upper_limit*(stds + 1e-5)
        lol = -upl
        cntsx = np.where(cntsx > upl,upl,cntsx)
        cntsx = np.where(cntsx < lol,lol,cntsx)        
        
        cnts = []
        for bp,sgn in self.break_pts:
            bp = bp.reshape([1,-1])
            if sgn == 1:
                cnts.append((cntsx > bp)* (cntsx - bp))
            else:
                cnts.append((cntsx < bp)* (bp - cntsx))
        cntsx = np.concatenate(cnts,axis = 1)
        ones = np.ones((ctrx.shape[0],1))
        y =  np.concatenate([ones,ctrx,cntsx,],axis = 1)
        if initially_flat_flag:
            return y.reshape([-1])
        return y
        
