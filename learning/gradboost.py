
import os
from learning.prods import N_TIME, get_clean_data,pick_time_index
from learning.genetic import gen_sol_location
import numpy as np
import dask.distributed
import xgboost as xgb
import sys

from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

HYPERPARAMS_FOLDER = 'hyper_param_logs'
def get_nnz_feats(ti,yi):
    loc = gen_sol_location(ti,yi)
    w = np.load(loc)
    return np.where(np.abs(w)!=0)[0]

def get_hyper_param_logs(ti,yi):
    root = os.path.abspath(HYPERPARAMS_FOLDER)
    if not os.path.exists(root):
        os.makedirs(root)
    filename = f'hp_t{ti}_y{yi}.json'
    return os.path.join(root,filename)
def get_bayes_optimizer(hplogs_file,pbounds = None,random_state = 0):
    optimizer = BayesianOptimization(f = None, 
            pbounds = pbounds, 
            verbose = 2, \
            random_state = random_state,\
            allow_duplicate_points = True)
    logger = JSONLogger(path=hplogs_file,reset=False)
    if os.path.exists(hplogs_file):
        load_logs(optimizer, logs=[hplogs_file])
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)   
    optimizer.set_gp_params(alpha=1e-3) 
    utility = UtilityFunction(kind="ei", xi=1e-2)
    return optimizer,utility

class HyperParamFunctor:
    def __init__(self,dflt_params = {},niter = 1,time_index = 0, y_index = 1,negate:bool = False,client = None,n_cv = 4):
        self.client = client
        self.dflt_params = dflt_params
        self.niter = niter
        self.negate = negate
        print(f'df =  get_clean_data()')
        df =  get_clean_data()#.partitions[:4]
        df = pick_time_index(df,time_index)
        ycols = ['Y1','Y2']
        inds = get_nnz_feats(time_index,y_index)
        xcols = np.array([c for c in df.columns if c not in ycols])
        xcols = xcols[inds]
        ycol = [ycols[y_index]]
        xinds = [df.columns.tolist().index(c) for c in xcols]
        yinds = [df.columns.tolist().index(c) for c in ycol]
        print(f'converting to dask_array')
        df_arr = df.to_dask_array(lengths = True)

        x_arr = df_arr[:,xinds]
        y_arr = df_arr[:,yinds]

        split_indices = np.array_split(np.arange(df_arr.shape[0]),n_cv,axis = 0)
        cv_indices = []
        for i in range(n_cv):
            tr = np.concatenate(split_indices[:i]+split_indices[i+1:])
            ts = split_indices[i]
            cv_indices.append((tr,ts))
            
        ddmats = []
        for i,(tr,ts) in enumerate(cv_indices):
            print(f'forming DaskDMatrix for cv #{i}')
            trset = xgb.dask.DaskDMatrix(client, x_arr[tr,:],y_arr[tr])
            tsset = xgb.dask.DaskDMatrix(client, x_arr[ts,:],y_arr[ts])
            ddmats.append((trset,tsset))
        self.ddmats = ddmats
    @property
    def integer_type_params(self,):
        return [
            'max_depth','num_boost_round'
        ]
    def to_integer_transform(self,**params):
        int_types = self.integer_type_params
        for key,val in params.items():
            if key in int_types:
                params[key]= int(val)
        return params
    def __call__(self,**params):
        params = self.to_integer_transform(**params)
        num_boost_round = params.pop('num_boost_round',50)
        outputs = []
        for random_state in range(self.niter):
            pparam = self.dflt_params.copy()
            pparam.update(params)
            pparam['random_state'] = random_state
            for tr,ts in self.ddmats:
                output = xgb.dask.train(self.client,pparam,\
                    tr,evals = ((tr,'train'),(ts,'test')),verbose_eval=False, num_boost_round=num_boost_round)
                outputs.append(output['history']['test']['rmse'])
        return_dicts = []
        for it,it_scrs in enumerate(zip(*outputs)):
            pparams = params.copy()
            pparams['num_boost_round'] = it          
            val = np.mean(it_scrs)
            if self.negate:
                val = -val
            return_dicts.append(dict(params = pparams,target = val))
        return return_dicts

def main():
    tiyi = int(sys.argv[1]) - 1
    ti = tiyi%N_TIME
    yi = tiyi//N_TIME
    yi = yi%2
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)
    params = {
        "objective": "reg:squarederror",
    }
    tune_params = {
        "eta" : (1e-5,1e-1),
        "gamma" : (0,1000),
        "num_boost_round" : (1,200),
        "max_depth" : (2,10),
        "min_child_weight" : (1,1e4),
        "subsample" : (0.1,1.),
        "colsample_bytree" : (0.1,1.),
    }
    hpf = HyperParamFunctor(dflt_params = params,negate=True,niter = 16,time_index=ti,y_index=yi,client =client,n_cv = 4)
    hplogs = get_hyper_param_logs(ti,yi)
    opt,uti = get_bayes_optimizer(hplogs,tune_params,0)
    while True:
        params = opt.suggest(uti)
        register_records = hpf(**params)
        for param_target in register_records:
            opt.register(**param_target)



if __name__ == '__main__':
    main()