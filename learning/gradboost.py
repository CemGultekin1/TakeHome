
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
XGB_PARAMS = {
    "eta" : (1e-5,1e-1),
    "gamma" : (0,1000),
    "num_boost_round" : (1,200),
    "max_depth" : (2,10),
    "min_child_weight" : (1,1e4),
    "subsample" : (0.1,1.),
    "colsample_bytree" : (0.1,1.),
}

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
    def __init__(self,dflt_params = {},niter = (0,1),test_run:bool = False,report_r2:bool = False,time_index = 0, y_index = 1,negate:bool = False,client = None,n_cv = 4):
        self.client = client
        self.dflt_params = dflt_params
        self.niter = niter
        self.negate = negate
        print(f'df =  get_clean_data()',flush = True)
        if test_run:
            df =  get_clean_data().partitions[:4]
        else:
            df =  get_clean_data()
        df = pick_time_index(df,time_index)
        ycols = ['Y1','Y2']
        inds = get_nnz_feats(time_index,y_index)
        xcols = np.array([c for c in df.columns if c not in ycols])
        xcols = xcols[inds]
        ycol = [ycols[y_index]]
        xinds = [df.columns.tolist().index(c) for c in xcols]
        yinds = [df.columns.tolist().index(c) for c in ycol]
        print(f'converting to dask_array',flush =True)
        df_arr = df.to_dask_array(lengths = True)

        x_arr = df_arr[:,xinds]
        y_arr = df_arr[:,yinds]

        inds = np.arange(df_arr.shape[0])
        rng = np.random.default_rng(0)
        rng.shuffle(inds)
        
        split_indices = np.array_split(inds,n_cv,axis = 0)
        cv_indices = []
        for i in range(n_cv):
            tr = np.concatenate(split_indices[:i]+split_indices[i+1:])
            ts = split_indices[i]
            cv_indices.append((tr,ts))
            
        ddmats = []
        self.sc2 = None
        if report_r2:
            self.sc2 = []
        for i,(tr,ts) in enumerate(cv_indices):
            print(f'forming DaskDMatrix for cv #{i}',flush=True)
            trset = xgb.dask.DaskDMatrix(client, x_arr[tr,:],y_arr[tr])
            tsset = xgb.dask.DaskDMatrix(client, x_arr[ts,:],y_arr[ts])
            ddmats.append((trset,tsset))                                
            if report_r2:
                self.sc2.append(np.mean(y_arr[ts]**2).compute())            
        self.ddmats = ddmats
        if report_r2:
            self.sc2 = np.array(self.sc2).reshape([-1,1])
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
    def __call__(self,**_params):
        params = self.dflt_params.copy()
        params1 = _params.copy()
        params.update(params1)
        params = self.to_integer_transform(**params)
        num_boost_round = params.pop('num_boost_round',50)
        outputs = np.zeros((num_boost_round,len(self.ddmats),self.niter[1] - self.niter[0]))
        for j,(tr,ts) in enumerate(self.ddmats):            
            for k,random_state in enumerate(range(self.niter[0],self.niter[1])):
                pparam = params.copy()
                pparam['random_state'] = random_state
                output = xgb.dask.train(self.client,pparam,\
                    tr,evals = ((tr,'train'),(ts,'test')),verbose_eval=False, num_boost_round=num_boost_round)
                outputs[:,j,k] = np.array(output['history']['test']['rmse'])
        return_dicts = []
        return_vals = []
        for it in range(outputs.shape[0]):
            rmse_scrs = outputs[it,...]
            pparams = params.copy()
            pparams['num_boost_round'] = it+1    
            mse = np.array(rmse_scrs)**2
            
            # if self.sc2 is not None:
            #     r2 = 1 - mse/self.sc2
            #     val = np.mean(r2)
            #     return_vals.append(r2)
            # else:
            val = np.sqrt(np.mean(mse))
            return_vals.append(mse)            
            if self.negate:
                val = -val
            return_dicts.append(dict(params = pparams,target = val))
        return return_vals,return_dicts


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
    
    hpf = HyperParamFunctor(dflt_params = params,negate=True,niter = (0,2),time_index=ti,y_index=yi,client =client,n_cv = 4)
    hplogs = get_hyper_param_logs(ti,yi)
    opt,uti = get_bayes_optimizer(hplogs,XGB_PARAMS,0)
    while True:
        params = opt.suggest(uti)
        _,register_records = hpf(**params)
        for param_target in register_records:
            opt.register(**param_target)



if __name__ == '__main__':
    main()