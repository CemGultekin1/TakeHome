
from typing import Any
import numpy as np
import xgboost as xgb
import dask.distributed
import warnings
warnings.filterwarnings("ignore")

class R2Metric:
    def __init__(self,sc2:float) -> None:
        self.sc2 = sc2
    def __call__(self, mse:float) -> Any:
        return 1- mse/self.sc2

class BlackBoxFunctor:
    def __init__(self,dask_dataset,xcols,ycol,n_cv,**xgb_params) -> None:
        cluster = dask.distributed.LocalCluster()
        self.client = dask.distributed.Client(cluster)
        self.xcols = xcols
        self.ycol = ycol
        self.xgb_params = xgb_params
        self.n_cv = n_cv
        self.metrics = []
        cv_splits = np.array_split(np.arange(dask_dataset.npartitions),n_cv)
        cv_datasets = []
        for i,inds in enumerate(cv_splits):
            dfsplit = dask_dataset.partitions[inds]
            sc2 = np.mean(np.power(dfsplit[ycol],2)).compute()
            self.metrics.append(R2Metric(sc2))            
            dsplit = xgb.dask.DaskDMatrix(self.client, dfsplit[self.xcols], dfsplit[self.ycol])
            cv_datasets.append((dsplit,f'part_{i}'))
        self.cv_datasets = cv_datasets
        
        
    def train_xgboost(self,dtrain,eval,metrics,**kwargs):
        params = self.xgb_params.copy()
        params.update(kwargs)
        output = xgb.dask.train(self.client,params,dtrain,evals=eval,num_boost_round = 200,early_stopping_rounds=20)
        it = output['booster'].best_iteration
        scr = []
        for metric,eval_dict in zip(metrics,output['history'].values()):
            scr.append(metric(eval_dict['rmse'][it]**2))
        # for metric in metrics:
        #     scr.append(metric(np.random.randn()**2))
        return scr
    def __call__(self,**kwargs):
        scr = []
        for i in range(self.n_cv):
            dtrain = self.cv_datasets[i][0]
            evals = self.cv_datasets.copy()
            evals.pop(i)
            submetrics = self.metrics.copy()
            submetrics.pop(i)
            scr.extend(self.train_xgboost(dtrain,evals,submetrics,**kwargs))
        return np.mean(scr)
            