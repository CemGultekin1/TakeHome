from learning.gradboost import HyperParamFunctor, get_bayes_optimizer,get_hyper_param_logs,XGB_PARAMS,N_TIME, get_nnz_feats
import itertools
import numpy as np
import dask.distributed
def main():
    # for ti,yi in itertools.product(range(N_TIME),range(2)):
    time_index,y_index = 0,1
    logs = get_hyper_param_logs(time_index,y_index)
    opt,_ = get_bayes_optimizer(logs,XGB_PARAMS)
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)
    
    params = {
        "objective": "reg:squarederror",
    }
    max_params = opt.max['params']
    max_params.update(params)
    print(f'(time_index,y_index) = {time_index,y_index},\n\t best params = {max_params}')
    hpf = HyperParamFunctor(dflt_params = max_params,\
                            client = client,\
                            report_r2=True,\
                            time_index=time_index,\
                            y_index=y_index,\
                            n_cv=8,\
                            niter=(3,32),\
                            test_run=False)
    outs,_ = hpf()
    mse = outs[-1]
    m = np.mean(mse)
    s = np.std(mse)
    print(mse)
    print(f' r2 = {m} +/- {s}')
    print(hpf.sc2)
    

if __name__ == '__main__':
    main()