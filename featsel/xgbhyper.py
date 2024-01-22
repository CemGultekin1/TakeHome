
import os
from featsel.constants import N_DAY_TIME,HYPER_PARAM_LOGS
from featsel.normaleqs import get_clean_data,pick_day_time_index
from featsel.genetic import gen_sol_location
import numpy as np
import dask.distributed
import xgboost as xgb
import sys

from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

XGB_PARAMS = {
    "eta" : (1e-4,5e-1),
    "gamma" : (0.1,1000),
    "num_boost_round" : (1,250),
    "max_depth" : (2,12),
    "min_child_weight" : (1,1e4),
    "subsample" : (0.5,1.),
    "colsample_bytree" : (0.5,1.),
    "lambda" : (1e-5,1e3),
    "alpha" : (1e-5,1e3),
}
LOG_SCALED_PARAMS = "eta gamma min_child_weight lambda alpha".split()

class LogTransform:
    @staticmethod
    def transform(_forward = False,**param):
        transform =  lambda  x: np.log10(x) if _forward else np.power(10.,x)
        pparam = param.copy()
        for key in pparam:
            if key in LOG_SCALED_PARAMS:
                if isinstance(pparam[key] ,tuple):
                    pparam[key] = tuple(transform(x) for x in pparam[key])
                else:
                    pparam[key] = transform(pparam[key])
        return pparam
    @staticmethod
    def forward(**param):
        return LogTransform.transform(_forward = True,**param)
    @staticmethod
    def backward(**param):
        return LogTransform.transform(_forward = False,**param)

        

def get_nnz_feats(ti,yi):
    loc = gen_sol_location(ti,yi,makedirs_permit=False)
    w = np.load(loc)
    if len(w) == 376:
        w = w[:-1]
    return np.where(np.abs(w)!=0)[0]

def get_hyper_param_logs(ti,yi):
    root = os.path.abspath(HYPER_PARAM_LOGS)
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
    def __init__(self,dflt_params = {},niter = (0,1),test_run:bool = False,day_time_index = 0, y_index = 1,client = None,n_cv = 4):
        self.client = client
        self.dflt_params = dflt_params
        self.niter = niter
        if test_run:
            df =  get_clean_data().partitions[:4]
        else:
            df =  get_clean_data()
        df = pick_day_time_index(df,day_time_index)
        ycols = ['Y1','Y2']
        inds = get_nnz_feats(day_time_index,y_index)
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
                
        split_indices = np.array_split(inds,n_cv,axis = 0)
        cv_indices = []
        for i in range(n_cv):
            tr = np.concatenate(split_indices[:i]+split_indices[i+1:])
            ts = split_indices[i]
            cv_indices.append((tr,ts))
            
        ddmats = []

        self.sc2 = np.mean(y_arr**2).compute()
        for i,(tr,ts) in enumerate(cv_indices):
            print(f'forming DaskDMatrix for cv #{i}',flush=True)
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
                outputs[:,j,k] = np.array(output['history']['test']['rmse'])**2
        return_dicts = []
        return_vals = []
        for it in range(outputs.shape[0]):
            mse = outputs[it,...]
            pparams = params.copy()
            pparams['num_boost_round'] = it+1
            r2 = 1 - np.mean(mse)/self.sc2
            return_vals.append(r2)            
            
            exkeys = list(pparams.keys())
            for key in exkeys:
                if key not in _params:
                    pparams.pop(key)    
                    
            return_dicts.append(dict(params = pparams,target = r2))
        return return_vals,return_dicts


def main():
    tiyi = int(sys.argv[1]) - 1
    ti = tiyi%N_DAY_TIME
    yi = (tiyi//N_DAY_TIME)%2
    print(f'day_time_index = {ti}, y_index = {yi}',flush = True)
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)
    params = {
        "objective": "reg:squarederror",
    }
    
    hpf = HyperParamFunctor(dflt_params = params,niter = (0,2),day_time_index=ti,y_index=yi,client =client,n_cv = 4)
    hplogs = get_hyper_param_logs(ti,yi)
    transformed_bounds = LogTransform.forward(**XGB_PARAMS)
    opt,uti = get_bayes_optimizer(hplogs,transformed_bounds,0)
    while True:
        params = opt.suggest(uti)
        pparams = LogTransform.backward(**params)
        _,register_records = hpf(**pparams)
        for param_target in register_records:
            param_target['params'] = LogTransform.forward(**param_target['params'])
            opt.register(**param_target)



if __name__ == '__main__':
    main()