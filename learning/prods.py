import sys
import numpy as np
import dask.distributed
import os
import warnings
import dask.dataframe as dataframe
warnings.filterwarnings("ignore")


ROOT_FOLDER = 'innerprods'
N_CV = 8
N_TIME = 4
PROD_TYPES = 'xx xy yy'.split()

def read_parquet():
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    return dataframe.read_parquet(parquet_files)

def prod_location(time_index,cvi,prodt):
    folder= os.path.join(ROOT_FOLDER,f't{time_index}p{N_TIME}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    assert prodt in PROD_TYPES
    filename = f'cv{cvi}_{prodt}.npy'
    return os.path.abspath(os.path.join(folder,filename))
def pick_time_index(df,t):
    reltime = df['reltime']
    df = df[reltime < t+0.5]
    df = df[reltime > t-0.5]
    df = df.drop(columns = ['Q1','Q2','reltime'])
    return df
    
def single_time_index(df,cvi,ti):
    df = pick_time_index(df,ti)
    ycols = ['Y1','Y2']
    xinds = [i for i,c in enumerate(df.columns) if c not in ycols]
    yinds = [df.columns.tolist().index(c) for c in ycols]
    
    df_arr = df.to_dask_array(lengths = True)
    
    inds = np.arange(df_arr.shape[0])
    rng = np.random.default_rng(0)
    rng.shuffle(inds)
    
    
    split_indices = np.array_split(inds,N_CV,axis = 0)
    lincomps = {}
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
        lincomps[i] = dict(
            xx = xx,xy = xy,yy =yy
        )
    
    
    for i,lc in lincomps.items():
        for c,v in lc.items():
            address = prod_location(ti,i,c)
            print(f'saving {address}')
            np.save(address.replace('.npy',''),v.compute())

def get_clean_data():
    df = read_parquet().select_dtypes('number')
    t = df['time']
    maxt = t.max()
    mint = t.min()
    relt = (t-mint)/(maxt-mint)
    relt = np.floor(relt*N_TIME)
    df['reltime'] = relt
    df = df.drop(columns = ['time'])
    df = df[df['Q1'] > 0.9999]
    df = df[df['Q2'] > 0.9999]
    df[df >= 999] = np.nan
    df[df <= -999] = np.nan
    df = df[np.sum(np.isnan(df),axis = 1)==0]
    return df
def main():
    t_cv = int(sys.argv[1]) - 1
    cvi = t_cv % N_CV
    ti = (t_cv//N_CV)%N_TIME
    cluster = dask.distributed.LocalCluster()
    _ = dask.distributed.Client(cluster)
    
    df = get_clean_data()
    for ti in range(N_TIME):
        single_time_index(df.copy(),cvi,ti)
        
if __name__ == '__main__':
    main()