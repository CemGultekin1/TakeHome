import os
import numpy as np
import dask
from distributed import Client
import shutil
from data.base import DataCleaner,INNER_PROD_FOLDER_NAME,read_parquet
from data.normalization import Normalization
from data.feature import Categoricals,CategoricalFeatureTransform,FEATURE_FOLDER_POINTER
from copy import copy
FILE_SAVE_TYPES = 'xx xy yy'.split()
BASE_NPY_KEY = f'_{FILE_SAVE_TYPES[0]}.npy'
TARGET_FOLDER = os.path.join(FEATURE_FOLDER_POINTER,INNER_PROD_FOLDER_NAME)
def get_statistics_path(time_partition):
    filename = [f'time_{time_partition[0]}_{time_partition[1]}_{key}.npy' for key in FILE_SAVE_TYPES]
    paths = [os.path.join(TARGET_FOLDER,f) for f in filename]
    return dict(tuple(zip(FILE_SAVE_TYPES,paths)))

def hierarchical_aggregation(files,time_partition,name :str= ''):    
    root = copy(TARGET_FOLDER)
    if name:
        root = os.path.join(root,name)
    if not os.path.exists(root):
        os.makedirs(root)
        
    split_files = np.array_split(np.array(files),len(files)/2)
    
    dfs = read_parquet()
    nrm = Normalization(n_type='vector',feat_transform=Categoricals().determine_ctgr())
    print(f'num features = {nrm.feat_transform.num_features}')
    print(f'nrm.is_available_in_file() = {nrm.is_available_in_file()}')
    if not nrm.is_available_in_file():
        nrm.compute_and_save_normalizations()
    kwargs = dict(allow_bad_x_density = 0.,allow_bad_y_density = 0.,fill_value = np.nan)
        
    k= 0
    path0 = os.path.join(root,f'h{k}')
    os.mkdir(path0) if not os.path.exists(path0) else None   

    @dask.delayed
    def outer_product_computations(i,file_group,**kwargs):
        filepath = os.path.join(path0,f'temp_{i}')
        print(filepath,flush = True)
        nrm = Normalization(n_type='vector',feat_transform=Categoricals().determine_ctgr())
        xydict = nrm.collect_normalizations_from_file()
        kwargs['xnormalizer'] = xydict['x']
        kwargs['feature_transform'] = nrm.feat_transform
        xx,xy,yy = get_values(file_group.tolist(),time_partition,dfs,**kwargs)
        values = dict( tuple(zip(FILE_SAVE_TYPES,(xx,xy,yy))))        
        for key,valu in values.items():
            if np.any(np.isnan(valu)):
                raise Exception(key + ' found nan values!')
        for key,val in values.items():
            np.save(filepath+f'_{key}',val)
        del values
        return 1
    
    output = []
    for i,file_group in enumerate(split_files):
        output.append(outer_product_computations(i,file_group,**kwargs))
        # break
    client = Client()
    client.compute(output,sync = True)
    folder = path0
    files = [fls.replace(BASE_NPY_KEY,'') for fls in os.listdir(folder) if BASE_NPY_KEY in fls]
    @dask.delayed
    def aggregation_inner_loop(path0,path1,i,file_group):
        vdict1 = {}
        for file1 in file_group:
            print(f'#{i}:\t{file1}',flush=True)
            vdict = {}
            for key in FILE_SAVE_TYPES:
                vdict[key] = np.load(os.path.join(path0,file1) + f'_{key}.npy')
            for key,val in vdict.items():
                if key not in vdict1:
                    vdict1[key] = val
                    continue
                vdict1[key] += val
        print(os.path.join(path1,f'temp_{i}'))
        for key in FILE_SAVE_TYPES:
            np.save(os.path.join(path1,f'temp_{i}_{key}'),vdict1[key])
        del vdict1
        return 1
    while len(files)>9:
        k+=1
        path1 = os.path.join(root,f'h{k}')
        os.mkdir(path1) if not os.path.exists(path1) else None
        split_files = np.array_split(np.array(files),len(files)/2)
        rlts = []
        for i,file_group in enumerate(split_files):
            rlts.append(aggregation_inner_loop(path0,path1,i,file_group))            
        client.compute(rlts,sync = True)        
        files = [fls.replace(BASE_NPY_KEY,'') for fls in os.listdir(path1) if BASE_NPY_KEY in fls]
        path0 = path1
    path1 = os.path.join(root,f'h{k}')
    files = [f.replace(BASE_NPY_KEY,'') for f in os.listdir(path1) if BASE_NPY_KEY in f]
    for key in FILE_SAVE_TYPES:
        reads = []
        for f in files:
            x = np.load(os.path.join(path1,f'{f}_{key}.npy'))
            reads.append(x)
        y = np.stack(reads,axis =0)
        np.save(root + f'_{key}.npy',y)
    del dfs
def clean_up(name:str):
    root = os.path.join(INNER_PROD_FOLDER_NAME,name)
    dirs = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root,f))]
    nums = [int(f[1]) for f in dirs]
    i = np.argmax(np.array(nums))
    fls = dirs[i]
    fls = [f for f in os.listdir(os.path.join(root,dirs[i])) if '.npy' in f]
    assert(len(fls) == 1)
    fl = fls[0]
    print(f'shutil.copy({os.path.join(root,dirs[i],fl)},\t\t{root + ".npy"})')
    shutil.copy(os.path.join(root,dirs[i],fl),root + ".npy")
    shutil.rmtree(root)
    
    
def get_values(t,time_partition,dfs,feature_transform : CategoricalFeatureTransform = None,xnormalizer = None,concat_flag = False,**kwargs):
    if isinstance(t,list):
        if concat_flag:
            xs = []
            ys = []
            remove_rows = kwargs.pop('remove_rows',True)
            for t_ in t:
                x,y = get_values(t_,time_partition,dfs,remove_rows = remove_rows,**kwargs)
                xs.append(x)
                ys.append(y)
            return np.concatenate(xs),np.concatenate(ys)
        np_type = np.longdouble
        num_feats = feature_transform.num_features
        xx = np.zeros((num_feats,num_feats),dtype = np_type)
        xy = np.zeros((num_feats,2),dtype = np_type)
        yy = np.zeros((2,2),dtype = np_type)
        for t_ in t:
            x,y = get_values(t_,time_partition,dfs,remove_rows = True,**kwargs)
            if np.any(np.isnan(x)):
                raise Exception(x)
            x = feature_transform(x)
            if np.any(np.isnan(x)):
                raise Exception(x)
            if xnormalizer is  not None:
                x = x @ xnormalizer            
            x,y = (s.astype(np_type) for s in (x,y))
            xx_ = x.T@x
            xy_ = (x.T@y).reshape([-1,2])
            yy_ = (y.T@y).reshape([2,2])
            xx += xx_
            xy += xy_
            yy += yy_
        return xx,xy,yy
        
    df = dfs.get_partition(t).compute()
    dc = DataCleaner(time_partition=time_partition,**kwargs)
    return dc(df)


def main():
    T = 4
    for tp in range(1,T):
        hierarchical_aggregation(np.arange(0,298),time_partition=(tp,T),name = f'time_{tp}_{T}')


if __name__ == '__main__':
    main()