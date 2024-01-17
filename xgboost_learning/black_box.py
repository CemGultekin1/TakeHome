
from typing import Any
import numpy as np
import xgboost as xgb
import dask.distributed
import warnings
warnings.filterwarnings("ignore")

class R2Metric:
    def __init__(self,sc2:float) -> None:
        self.sc2 = sc2
    def __call__(self, rmse:float) -> Any:
        return 1- rmse**2/self.sc2

class RMSE:
    def __call__(self, rmse:float) -> Any:
        return rmse

class BlackBoxFunctor:
    def __init__(self,dask_dataset,xcols,ycol,n_fold,n_iter,**xgb_params) -> None:
        print(f'cluster = dask.distributed.LocalCluster()',flush = True)
        cluster = dask.distributed.LocalCluster()
        self.client = dask.distributed.Client(cluster)
        self.xcols = xcols
        self.ycol = ycol
        self.n_iter = n_iter
        self.xgb_params = xgb_params
        self.n_fold = n_fold       
        self.dtrain = xgb.dask.DaskDMatrix(self.client, dask_dataset[self.xcols], dask_dataset[self.ycol])
        
        
    def train_xgboost(self,dtrain,eval,metrics,**kwargs):
        params = self.xgb_params.copy()
        params.update(kwargs)
        output = xgb.dask.cv(self.client,params,dtrain,num_boost_round = 200,early_stopping_rounds=20,nfold = self.n_fold)
        print(output)
        it = output['booster'].best_iteration
        scr = []
        for metric,eval_dict in zip(metrics,output['history'].values()):
            scr.append(metric(eval_dict['rmse'][it]))

        return scr
    def __call__(self,**kwargs_):
        kwargs = kwargs_.copy()
        scr = []
        for i in range(self.n_cv):
            dtrain = self.cv_datasets[i][0]
            evals = self.cv_datasets.copy()
            evals.pop(i)
            submetrics = self.metrics.copy()
            submetrics.pop(i)
            for seed in range(self.n_iter):
                kwargs['random_state'] = seed
                scr.extend(self.train_xgboost(dtrain,evals,submetrics,**kwargs))
        return np.mean(scr)
            