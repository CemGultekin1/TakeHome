import itertools
import sys
import numpy as np
import dask.distributed
import os
import warnings
import dask.dataframe as dataframe
warnings.filterwarnings("ignore")
from calibration.constants import NORMAL_EQS_FOLDER,N_CV,N_DAY_TIME,NORMAL_EQ_COMPS,PARQUET_DATA_PATH
"""
    Before selecting features using cross-validation (CV), it separates the data
    into N_CV many equal sized blocks in time and gathers normal equations from each block. 
"""

def read_parquet(PARQUET_DATA_PATH :str = PARQUET_DATA_PATH)->dask.dataframe:
    """
        Reads the whole parquet directory in the workspace.
        Args:
            "PARQUET_DATA_PATH"     : default defined in constants.py
    """
    parquet_files = [os.path.join(PARQUET_DATA_PATH, f) for f in os.listdir(PARQUET_DATA_PATH) if f.endswith('.parquet')]
    return dataframe.read_parquet(parquet_files)

def normal_eq_location(day_time_index:int,cvi:int,normal_eqt:str):
    """
        Returns the abspath to the .npy file storing the normal equations
        Args:
            "day_time_index" : specifies the time partition of the day out of "N_DAY_TIME" groups
            "cvi"        : cross validation group index
            "normal_eqt" : type of the normal equation component 
    """
    assert normal_eqt in NORMAL_EQ_COMPS
    
    folder= os.path.join(NORMAL_EQS_FOLDER,f't{day_time_index}p{N_DAY_TIME}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f'cv{cvi}_{normal_eqt}.npy'
    return os.path.abspath(os.path.join(folder,filename))
def pick_day_time_index(df:dask.dataframe,ti:int):
    """
        Removes the rows of "df" that doesn't belong to 
        a specific day time index. 
        Args:
            "df"    : dataframe with reltime specifying relative time in the day
            "t"     : specific index out of N_DAY_TIME 
    """
    reltime = df['reltime']
    df = df[reltime < ti+0.5]
    df = df[reltime > ti-0.5]
    df = df.drop(columns = ['Q1','Q2','reltime'])
    return df
    
def compute_normal_eqs(df:dask.dataframe,cvi:int,ti:int):
    """
    Divides "df" across days into N_CV many disjoint blocks. 
    For the particular block with index "cvi" computes(lazy) the
    normal equations for day time index "ti".
    Args:
        "df"    : data
        "cvi"   : cross-validation index out of N_CV
        "ti"    : day time index out of N_DAY_TIME
    """
    df = pick_day_time_index(df,ti)
    ycols = ['Y1','Y2']
    xinds = [i for i,c in enumerate(df.columns) if c not in ycols]
    yinds = [df.columns.tolist().index(c) for c in ycols]
    
    df_arr = df.to_dask_array(lengths = True)
    
    inds = np.arange(df_arr.shape[0])
    
    
    split_indices = np.array_split(inds,N_CV,axis = 0)
    normal_eqs = {}
    for i,sp in enumerate(split_indices):
        if i != cvi:
            continue
        dfsp = df_arr[sp,:]
        x = dfsp[:,xinds]
        y = dfsp[:,yinds]
        x = np.where(np.isnan(x),0,x)
        y = np.where(np.isnan(y),0,y)
        xx = x.T@x
        xy = x.T@y
        yy = y.T@y
        normal_eqs = dict(
            xx = xx,xy = xy,yy =yy
        )
    return normal_eqs
    

def clean_raw_data()->dask.dataframe:
    """
        Reads the whole dataset and cleans it (lazy)
    """
    df = read_parquet().select_dtypes('number')
    t = df['time']
    maxt = t.max()
    mint = t.min()
    relt = (t-mint)/(maxt-mint)
    relt = np.floor(relt*N_DAY_TIME)
    df['reltime'] = relt
    df = df.drop(columns = ['time'])
    df = df[df['Q1'] > 0.9999]
    df = df[df['Q2'] > 0.9999]
    df[df >= 999] = np.nan
    df[df <= -999] = np.nan
    df = df[np.sum(np.isnan(df),axis = 1)==0]
    return df
def main():
    """
        Computes normal equations across N_CV many time blocks across the whole data.
        Each normal equation is specific to a day time index out of N_DAY_TIME parts. 
    """    
    cluster = dask.distributed.LocalCluster()
    _ = dask.distributed.Client(cluster)
    
    df = clean_raw_data()    
    for cvi,ti in itertools.product(range(N_CV),range(N_DAY_TIME)):
        print(f'cross_val_index #{cvi}, day_time_index #{ti}',flush = True)
        normal_eqs = compute_normal_eqs(df.copy(),cvi,ti)
        for _type,_dask_arr in normal_eqs.items():
            address = normal_eq_location(ti,cvi,_type)
            print(f'saving {address}',flush = True)
            np.save(address.replace('.npy',''),_dask_arr.compute())
        
if __name__ == '__main__':
    main()