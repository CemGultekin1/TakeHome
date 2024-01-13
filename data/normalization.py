from data.base import DataCleaner,read_parquet
from data.feature import FeatureInfo,CategoricalFeatureTransform
import numpy as np
import os

def na_skip_avg(z):
    zs = np.sum(np.where(np.isnan(z),0,z),axis = 0)
    cs = np.sum(np.where(np.isnan(z),0,1),axis = 0)
    return zs/cs

class Normalization(FeatureInfo):
    def __init__(self,n_type:str = 'standard',feat_transform:CategoricalFeatureTransform = None) -> None:
        super().__init__()
        self.n_type = n_type
        self.feat_transform = feat_transform
    def compute_and_save_normalizations(self,):
        fun = self.__getattribute__(self.n_type)
        xy = self.get_data()
        
        
        # xx = xy['x']
        # import matplotlib.pyplot as plt
        # mxx = np.mean(np.abs(xx),axis = 0)
        # plt.plot(mxx)
        # plt.savefig('mxx.png')
        # raise Exception(mxx.shape)
        
        savedict = fun(xy)
        self.write2file(**savedict)
    def is_available_in_file(self,):
        files = [f for f in os.listdir(self.save_dir) if self.n_type in f]
        return bool(files)
    def absolute(self,xy):
        savedict = {}
        for key,z in xy.items():
            mz = na_skip_avg(np.abs(z))
            savedict[key] = mz
        return savedict
    def collect_normalizations_from_file(self,): 
        files = [f for f in os.listdir(self.save_dir)  if '.npy' in f and self.n_type in f]
        dictform = {}
        for f in files:            
            path = os.path.join(self.save_dir,f)
            key = f.split('_')[-1].replace('.npy','')
            dictform[key] = np.load(path,allow_pickle=False)#[()]
        return dictform
    @property
    def path(self,):
        return  os.path.join(self.save_dir,self.n_type)
    def write2file(self,**kwargs):
        for key,val in kwargs.items():   
            assert isinstance(val,np.ndarray)  
            np.save(self.path + '_'+key,val)
    def get_data(self,):
        dfs = read_parquet()
        df = dfs.get_partition(self.train_parquet_files[0])
        cdt = DataCleaner(allow_bad_x_density=0,allow_bad_y_density=0,fill_value=np.nan,remove_rows=True)
        x,y =cdt(df)        
        del dfs
        if self.feat_transform is not None:
            x = self.feat_transform(x)
        return dict(x = x,y = y)
    def standard(self,xy):                
        savedict = {}
        for key,z in xy.items():
            mz = na_skip_avg(z)
            sz = na_skip_avg(np.square(z)) - np.square(mz)
            savedict[key] = dict(
                avg = mz,
                std = sz
            )
        return savedict
    def vector(self,xy):
        savedict = {}
        for key,z in xy.items():            
            _,s,vh = np.linalg.svd(z)
            sinv = np.where(s/s[0] < 1e-3, 1,1/s)
            savedict[key] = vh.T@np.diag(sinv)
            # print(f'savedict[{key}] = {savedict[key].shape}')
        return savedict
        
    