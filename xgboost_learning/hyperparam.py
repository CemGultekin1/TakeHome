from data.base import read_parquet
import pandas
import numpy as np

from bayes_opt import BayesianOptimization, UtilityFunction
import warnings
warnings.filterwarnings("ignore")

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import os 
from xgboost_learning.black_box import BlackBoxFunctor

def get_clean_data(nparts):
    df = read_parquet().select_dtypes('number').partitions[:nparts]
    cols = df.columns
    xcols = [c for c in cols if 'X' in c]
    ycols = [c for c in cols if 'Y1' in c]
    maxtime = df['time'].max().compute()
    mintime = df['time'].min().compute()
    ntime = 4
    df = df.assign(reltime = lambda x: np.floor((x['time'] - mintime)/(maxtime - mintime)*ntime))
    df = df.drop(columns = ['time'])
    def clean(row:pandas.Series):
        row.loc[np.abs(row) > 999] = np.nan
        if row['Q1'] < 0.99999 or np.any(np.isnan(row[ycols])) or np.mean(np.isnan(row)) > 0.3:
            row.iloc[:] = np.nan              
        row = row.drop(columns=['Q1'])        
        return row
    df = df.apply(clean,axis = 1,by_row=False).fillna(0)
    return df,xcols,ycols

class BayesSearchParams:
    def __init__(self,**kwargs) -> None:
        self.params = {}
        self.transforms = {}
        for key,val in kwargs.items():
            if not isinstance(val,tuple):
                self.params[key] = val
                continue
            self.params[key] = val[0]            
            name = None
            if len(val) > 2:
                name = val[2]
            self.transforms[key] = (val[1],name)
    def __call__(self,**kwargs):
        keys =list(kwargs.keys())
        new_kwargs = {}
        for key in keys:
            val = kwargs[key]
            if key in self.transforms:
                transform,name = self.transforms[key]
                val = transform(val)
                if name is not None:
                    key = name
                else:
                    kwargs[key] = val
                new_kwargs[key] = val
            else:
                new_kwargs[key] = val
        return kwargs,new_kwargs
    @property
    def pbounds(self,):
        return self.params

def main():
    nfiles = 2
    ncv = 2
    niter = 10
    df,xcols,_ = get_clean_data(nfiles)

    param_transform_pairs = dict(
        log_eta = ([-5,0],lambda x: np.exp(x),'eta'),
        gamma = [0,8],
        max_depth = ([2,12],lambda x:int(x)),    
        log_colsample_bytree = ([-3,0],lambda x: np.exp(x),'colsample_bytree'),
        log_colsample_bylevel = ([-3,0],lambda x: np.exp(x),'colsample_bylevel'),
        log_colsample_bynode = ([-3,0],lambda x: np.exp(x),'colsample_bynode'),
    )
    bsp = BayesSearchParams(**param_transform_pairs)
    

    params = {"objective": "reg:squarederror"}
    bbf = BlackBoxFunctor(df,xcols,['Y1'],ncv,niter,**params)
    optimizer = BayesianOptimization(f = None, 
                                    pbounds = bsp.pbounds, 
                                    verbose = 2, random_state = 0)
    log_file = 'bayesian_log'
    if not os.path.exists(log_file):
        logger = JSONLogger(path=log_file,reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    else:
        load_logs(optimizer, logs=[log_file])
        logger = JSONLogger(path=log_file,reset=False)
    utility = UtilityFunction(kind = "ucb", kappa_decay= 0.95, xi = 0.01,kappa_decay_delay=20)
    while True:
        params = optimizer.suggest(utility)
        params,transformed_params = bsp(**params)
        r2 = bbf(**transformed_params)
        try:
            optimizer.register(params = params, target = r2)
        except:
            pass

        

if __name__ == '__main__':
    main()