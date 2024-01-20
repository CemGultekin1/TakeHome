from data.base import read_parquet
import pandas
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import os 
from xgboost_learning.black_box import BlackBoxFunctor

def get_clean_data(beg,end,ynum):
    df = read_parquet().select_dtypes('number').partitions[beg:end]
    cols = df.columns
    xcols = [c for c in cols if 'X' in c]
    ycols = [c for c in cols if f'Y{ynum}' in c]
    df = df[df['Q1'] > 0.9999]
    df = df[df['Q2'] > 0.9999]
    t = df['time']
    maxt = t.max()
    mint = t.min()
    relt = (t-mint)/(maxt-mint)
    relt = np.floor(relt*4)
    df['reltime'] = relt
    df = df.drop(columns = ['time'])
    df[np.abs(df) >= 999] = 0
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
power_of_ten = lambda x:np.power(10,x)
round_to_int = lambda x:int(x)
param_transform_pairs = dict(
    log10_eta = ([-4,0],power_of_ten,'eta'),
    gamma = [0,1000],
    num_boost_round = ([10,200],round_to_int),
    max_depth = ([2,10],round_to_int),    
    log10_alpha = ([-5,1],power_of_ten,'alpha'),
    log10_lambda = ([-5,1],power_of_ten,'lambda'),
    log10_min_child_weight = ([0,5],lambda x:round_to_int(power_of_ten(x)),'min_child_weight'),        
    log10_subsample = ([-1,0],power_of_ten,'subsample'),
    log10_colsample_bytree = ([-1,0],power_of_ten,'colsample_bytree')
)
def main():
    nfiles = 128
    ncv = 2
    niter = 1
    index = int(sys.argv[1])
    ynum = (index - 1) % 2 + 1
    random_state = index//2
    print(f'nfiles = {nfiles} ncv = {ncv} niter = {niter} ynum = {ynum}, random_state = {random_state}',flush = True)
    df,xcols,_ = get_clean_data(0,nfiles,ynum)
    
    bsp = BayesSearchParams(**param_transform_pairs)

    params = {"objective": "reg:squarederror"}
    bbf = BlackBoxFunctor(df,xcols,[f'Y{ynum}'],ncv,niter,**params)
    optimizer = BayesianOptimization(f = None, 
                                    pbounds = bsp.pbounds, 
                                    verbose = 2, \
                                    random_state = random_state,\
                                    allow_duplicate_points = True)
    log_file = os.path.abspath(f'bayes_logs1/bayesian_y{ynum}_rnd{random_state}')
    logger = JSONLogger(path=log_file,reset=False)
    if os.path.exists(log_file):
        load_logs(optimizer, logs=[log_file])
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    utility = UtilityFunction(kind="ei", xi=1e-4)#kind = "ucb", kappa_decay= 0.95, xi = 0.01,kappa_decay_delay=20)
    while True:
        params = optimizer.suggest(utility)
        params,transformed_params = bsp(**params)
        negrmse_by_num_iter = bbf(**transformed_params)
        try:
            for key,val in negrmse_by_num_iter.items():
                pp = params.copy()
                pp['num_boost_round'] = key
                optimizer.register(params = pp, target = val)
        except:
            pass

        

if __name__ == '__main__':
    main()