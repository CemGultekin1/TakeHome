import dask.dataframe as dd
import pandas
from torch.utils.data import Dataset,get_worker_info
import numpy as np
from data.base import DataCleaner,BaseData
from data.normalization import Normalization
from data.feature import Categoricals
import logging
from deep_learning.infeats import SelectedFeatures
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

class PriceTransform(BaseData):
    def __init__(self,ntime_daily =4,return_order :str = 'x,y'):
        super().__init__()
        self._normalization = None
        self.ntime_daily = ntime_daily
        self.return_order = return_order
        self.feature_transform = DataCleaner(allow_bad_x_density=0.3,allow_bad_y_density=0.,remove_rows=False,fill_value=np.nan)    
        self.data_cleaner = self.feature_transform                        
    @property
    def normalization_consts(self,):
        if self._normalization is None:
            norm = Normalization(n_type = 'absolute',\
                feat_transform = None)
            if not norm.is_available_in_file():
                norm.compute_and_save_normalizations()
            normdict = norm.collect_normalizations_from_file()
            self._normalization = tuple(normdict.values())
        return self._normalization
    @property
    def output_dimensions(self,):
        dims_by_name = dict(
            t = self.ntime_daily,\
            m = len(self.xcols),\
            x = self.feature_transform.num_features,\
            y = len(self.ycols))
        dims = []
        for ss in self.return_order.split(','):
            n= 0
            for s in ss:
                n += dims_by_name[s]
            dims.append(n)
        return tuple(dims)
    def inverse_transform(self,ys):
        _,ya = self.normalization_consts
        return ys*ya
    def get_time_features(self,sample:pandas.DataFrame):
        rts = self.data_cleaner.relative_time_value(sample)
        seg = np.minimum(np.floor(rts*self.ntime_daily).astype(int),self.ntime_daily - 1)
        dt = np.zeros((rts.shape[0],self.ntime_daily),dtype = float)
        dt[np.arange(dt.shape[0]),seg] = 1.
        return dt,seg
    def __call__(self,sample):
        t,tis = self.get_time_features(sample)
        x,y = self.feature_transform(sample)  
        # xs = []
        # for xi,ti in zip(x,tis):
        #     yi = self.feature_transform(xi,ti)
        #     xs.append(yi)
        # x = np.stack(xs,axis = 0)
        xa,ya = self.normalization_consts
        x = x/xa
        y = y/ya
        m = np.isnan(x)
        x = np.where(np.isnan(x),0,x)
        ins = dict(x = x,t = t,m = m,y = y)
        vecs = [np.concatenate([ins[s] for s in ss],axis = 1) for ss in self.return_order.split(',')]
        return vecs

class CustomDataset(Dataset):
    def __init__(self, parquet_files,ncpu, per_request = 64,internal_shuffle_flag = True,transform=None):
        """
        Args:
            file_paths (list): List of file paths.
            split_ratio (tuple): A tuple representing the split ratios for train, validation, and test sets.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.parquet_files = parquet_files
        self._select_parques = None
        self.ncpu = ncpu        
        self.per_request = per_request
        
        self.num_files = len(self.parquet_files)
        
        self.this_file = None
        self.in_file_shuffle = None
        
        self.sampling_rate = 1.
        self.per_file_lines = 22500
        self.tot_num_samples = np.floor(self.num_files*self.per_file_lines*self.sampling_rate ).astype(int)
              
        self.num_files_per_batch = 0        
        self.internal_shuffle_flag = internal_shuffle_flag
        
        self.internal_counter = self.internal_counter_lim
    @property
    def internal_counter_lim(self,):       
        return int(self.num_files_per_batch*self.per_file_lines* self.sampling_rate)
    @property
    def worker_idx(self,):
        if self.ncpu > 0:
            widx = get_worker_info().id
        else:
            widx = 0
        return widx
    @property 
    def select_parques(self,):
        if self._select_parques is None:
            splits = np.array_split(self.parquet_files,max(self.ncpu,1))
            self._select_parques = splits[self.worker_idx].tolist()
        return self._select_parques
    def rotate_files(self,offset):
        if offset % len(self.select_parques) != 0:
            offset = offset % len(self.select_parques)
            self._select_parques = self._select_parques[offset:] +\
                            self._select_parques[:offset]
    def update_file(self,):           
        self.num_files_per_batch = min(len(self.select_parques),30)    
        parqs = self.select_parques[:self.num_files_per_batch]
        self.this_file = dd.read_parquet(parqs).compute().select_dtypes('number')
        self.rotate_files(self.num_files_per_batch)
        self.in_file_shuffle = np.random.choice(\
                len(self.this_file),self.internal_counter_lim,\
                    replace = False)
        
    def __len__(self):
        return int(np.ceil(self.tot_num_samples/self.per_request))
    def __getitem__(self, *args):
        if self.internal_counter >= self.internal_counter_lim:
            self.internal_counter = 0
            self.update_file()
        # print(f'#{self.worker_idx} = {self.internal_counter}')
        i0 = self.internal_counter
        i1 = self.internal_counter + self.per_request
        i1 = min(self.internal_counter_lim,i1)
        ix = slice(i0,i1)
        if self.internal_shuffle_flag:
            ixs = self.in_file_shuffle[ix]
            sample = self.this_file.iloc[ixs]
        else:
            sample = self.this_file.iloc[ix]
        if self.transform:
            sample = self.transform(sample)
        self.internal_counter+=self.per_request
        if self.per_request == 1:
            sample = tuple(s.reshape([-1,]) for s in sample)
        return sample



# class CustomDataset(Dataset):
#     def __init__(self, parquet_files,ncpu, internal_shuffle_flag = True,transform=None):
#         """
#         Args:
#             file_paths (list): List of file paths.
#             split_ratio (tuple): A tuple representing the split ratios for train, validation, and test sets.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.transform = transform
#         self.parquet_files = parquet_files
#         self._select_parques = None
#         self.ncpu = ncpu        
        
#         self.num_files = len(self.parquet_files)
        
#         self.this_file = None
#         self.in_file_shuffle = None
        
#         self.sampling_rate = 0.1
#         self.per_file_lines = 22500
#         self.tot_num_samples = np.floor(self.num_files*self.per_file_lines*self.sampling_rate ).astype(int)
              
#         self.num_files_per_batch = 0        
#         self.internal_shuffle_flag = internal_shuffle_flag
        
#         self.internal_counter = self.internal_counter_lim
#     @property
#     def internal_counter_lim(self,):       
#         return int(self.num_files_per_batch*self.per_file_lines* self.sampling_rate)
#     @property
#     def worker_idx(self,):
#         if self.ncpu > 0:
#             widx = get_worker_info().id
#         else:
#             widx = 0
#         return widx
#     @property 
#     def select_parques(self,):
#         if self._select_parques is None:
#             splits = np.array_split(self.parquet_files,max(self.ncpu,1))
#             self._select_parques = splits[self.worker_idx].tolist()
#         return self._select_parques
#     def rotate_files(self,offset):
#         if offset % len(self.select_parques) != 0:
#             offset = offset % len(self.select_parques)
#             self._select_parques = self._select_parques[offset:] +\
#                             self._select_parques[:offset]
#     def update_file(self,):              
#         self.num_files_per_batch = min(len(self.select_parques),5)    
              
#         self.this_file = dd.read_parquet(self.select_parques[:self.num_files_per_batch])   
#         self.this_file = self.this_file.select_dtypes('number').compute()  
        
#         self.rotate_files(self.num_files_per_batch)
#         self.in_file_shuffle = np.random.choice(\
#                 len(self.this_file),self.internal_counter_lim,\
#                     replace = False)
        
#     def __len__(self):
#         return self.tot_num_samples
#     def __getitem__(self, *args):
#         if self.internal_counter == self.internal_counter_lim:
#             self.internal_counter = 0
#             self.update_file()
#         if self.internal_shuffle_flag:
#             sample = self.this_file.iloc[self.in_file_shuffle[self.internal_counter]]
#         else:
#             sample = self.this_file.iloc[self.internal_counter]
#         if self.transform:
#             sample = self.transform(sample)
#         self.internal_counter+=1
#         return sample