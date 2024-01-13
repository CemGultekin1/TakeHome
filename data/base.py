from typing import Union, Tuple
import numpy as np
from pandas import DataFrame as pd_df
import os
import dask.dataframe as dataframe
from dask.dataframe import DataFrame as d_df

INNER_PROD_FOLDER_NAME = 'inner_prods'

def read_parquet():
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    return dataframe.read_parquet(parquet_files)

class BaseData:
    def __init__(self) -> None:
        self.xcols = [f'X{i}' for i in range(1,376)]
        self.ycols = [f'Y{i}' for i in range(1,3)]
        self.qcols = [f'Q{i}' for i in range(1,3)]

class DataCleaner(BaseData):
    def __init__(self,time_partition:Tuple[int,int] = (0,1),\
                allow_bad_x_density :float= 0.3,\
                allow_bad_y_density :float= 0.,\
                remove_rows:bool = False,\
                fill_value:float = np.nan,\
                time_extremums :Tuple[int,int] = (35101000,57600000),
                ) -> None:
        super().__init__()
        self.time_partition = time_partition
        self.allow_bad_x_density = allow_bad_x_density 
        self.allow_bad_y_density = allow_bad_y_density
        self.remove_rows = remove_rows
        self.fill_value = fill_value    
        self.time_extremums = time_extremums        
    def relative_time_value(self,df:Union[pd_df,d_df]):
        ts = np.array(df[['time']].values)
        rts = (ts - self.time_extremums[0])/(self.time_extremums[1] - self.time_extremums[0])
        return rts.reshape([-1,])
    def apply_masks(self,df:Union[pd_df,d_df]):
        rts = self.relative_time_value(df)
        ti,T = self.time_partition
        rts = np.floor(rts*T).astype(int)
        tmask = rts == ti

        
        x = np.array(df[self.xcols].values).reshape([len(rts),-1])
        x[np.abs(x)>=9999] = np.nan
        bad_x = np.isnan(x)
        xmask = np.mean(bad_x,axis = 1) <= self.allow_bad_x_density
        
        y = np.array(df[self.ycols].values).reshape([len(rts),-1])
        y[np.abs(y)>=9999] = np.nan
        q = np.array(df[self.qcols].values).reshape([len(rts),-1])
        y = np.where(q > 0.99999,y,np.nan)
        bad_y = np.isnan(y)
        ymask = np.mean(bad_y,axis = 1) <= self.allow_bad_y_density        
        row_mask = xmask & ymask & tmask
        return x,y,row_mask
    
    def __call__(self,df:pd_df):
        x,y,namask = self.apply_masks(df)
        if self.remove_rows:
            x = x[namask,:]
            y = y[namask,:]
        return x,y
    