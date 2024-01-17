
import numpy as np
import xgboost as xgb
import dask.distributed
import warnings
warnings.filterwarnings("ignore")

class BlackBoxFunctor:
    def __init__(self,dask_dataset,xcols,ycol,n_cv,**xgb_params) -> None:
        cluster = dask.distributed.LocalCluster()
        self.client = dask.distributed.Client(cluster)
        self.xcols = xcols
        self.ycol = ycol
        self.xgb_params = xgb_params
        self.n_cv = n_cv
        cv_splits = np.array_split(np.arange(dask_dataset.npartitions),n_cv)
        cv_datasets = []
        for i,inds in enumerate(cv_splits):
            dfsplit = dask_dataset.partitions[inds]
            dsplit = xgb.dask.DaskDMatrix(self.client, dfsplit[self.xcols], dfsplit[self.ycol])
            cv_datasets.append((dsplit,f'part_{i}'))
        self.cv_datasets = cv_datasets
        
        
    def train_xgboost(self,dtrain,eval,**kwargs):
        params = self.xgb_params.copy()
        params.update(kwargs)
        output = xgb.dask.train(self.client,params,dtrain,evals=eval,num_boost_round = 200,early_stopping_rounds=20)
        it = output['booster'].best_iteration
        rmses = 0
        for eval_dict in output['history'].values():
            rmses+=eval_dict['rmse'][it]      
        return rmses/(self.n_cv - 1)
    def __call__(self,**kwargs):
        rmses = 0
        for i in range(self.n_cv):
            dtrain = self.cv_datasets[i][0]
            evals = self.cv_datasets.copy()
            evals.pop(i)
            rmses += self.train_xgboost(dtrain,evals,**kwargs)
        return rmses/self.n_cv
            