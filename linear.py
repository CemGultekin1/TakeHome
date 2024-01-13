from data.statistics import read_parquet,get_values
import numpy as np
import matplotlib.pyplot as plt
from genetic import Dataset
def reduced_rank_linear_systems_demo():
    dfs = read_parquet()
    numf = 1
    xts,yts = get_values(list(range(numf)),(0,1),dfs,remove_rows = True,concat_flag=True)
    xtr,ytr = get_values(list(range(100,100 + numf)),(0,1),dfs,remove_rows = True,concat_flag=True)
    del dfs
    
    u,s,vh = np.linalg.svd(xtr,full_matrices=False)
    svd_rel_err = np.linalg.norm(u@np.diag(s)@vh - xtr)/np.linalg.norm(xtr)
    print(f'svd_rel_err = {svd_rel_err}')
    r2s = []
    ranks = np.arange(100)
    for rank in ranks:
        sinv = np.concatenate([1/s[:rank],s[rank:]*0])
        w = vh.T@np.diag(sinv)@(u.T@ytr)
        ypred = xts @ w
        err = np.sum(np.square(yts - ypred),axis = 0)
        sc2 = np.sum(np.square(yts),axis = 0)        
        r2 = 1 - err/sc2
        r2s.append(r2.tolist() + [rank])
    r20,r21,rank = tuple(zip(*r2s))
    plt.plot(rank,r20,label = 'Y0',marker = 'x',linestyle = 'None')
    plt.plot(rank,r21,label = 'Y1',marker = '+',linestyle = 'None')
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.legend()
    plt.ylabel(f'R$^2$')
    plt.xlabel('rank')
    plt.ylim([-2e-1,2e-1])
    
    plt.savefig('rank_vs_r2.png')
    


def main():
    dfs = read_parquet()
    numf = 5
    time_partition = (0,1)
    
    kwargs = dict(fill_value = 0,allow_bad_x_density = 0.,remove_rows = False)
    
    xts,yts = get_values(list(range(numf)),time_partition,dfs,concat_flag=True,**kwargs)
    xtr,ytr = get_values(list(range(100,100 + numf)),time_partition,dfs,concat_flag=True,**kwargs)
        
    del dfs
    from data.feature import Categoricals
    transform = Categoricals().determine_ctgr()
    xtr_ctgr = xtr[:,transform.ctgr_flag]
    ctgr = np.where(xtr_ctgr == 0,1,0)
    # print(ctgr.shape)

    ts = (np.arange(ctgr.shape[0])/25000)%1
    q,r = np.linalg.qr(ctgr)
    arbs = np.abs(np.diag(r))
    inds, = np.where(arbs/np.amax(arbs) > 1e-3)
    print(len(inds))
    # uctgrs = uctgrs[:2500*10,:].T.reshape([158,-1,2500]).transpose([1,0,2]).reshape([-1,2500])
    # plt.imshow(uctgrs)
    # plt.savefig('ctgr.png')
    return
    xtr = transform(xtr)
    xts = transform(xts)
    
    mx = np.mean(np.abs(xtr),axis = 0).reshape([1,-1]) + 1e-9
    my = np.mean(np.abs(ytr),axis = 0).reshape([1,-1]) + 1e-9
    xtr = xtr/mx
    xts = xts/mx
    ytr = ytr/my
    yts = yts/my
    
    assert not np.any(np.isnan(xtr))
    assert not np.any(np.isnan(ytr))
    u,s,vh = np.linalg.svd(xtr.T@xtr,full_matrices=False)
    sinv = np.where(s/s[0] > 1e-3,1/s,0)
    xy = xtr.T@ytr
    r2s = []
    ranks = np.arange(100)
    for rank in ranks:
        sinv = np.concatenate([1/s[:rank],s[rank:]*0])
        w = vh.T@np.diag(sinv)@(u.T@xy)
        ypred = xts @ w
        err = np.sum(np.square(yts - ypred),axis = 0)
        sc2 = np.sum(np.square(yts),axis = 0)        
        r2 = 1 - err/sc2
        r2s.append(r2.tolist() + [rank])
    r20,r21,rank = tuple(zip(*r2s))
    plt.plot(rank,r20,label = 'Y0',marker = 'x',linestyle = 'None')
    plt.plot(rank,r21,label = 'Y1',marker = '+',linestyle = 'None')
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.legend()
    plt.ylabel(f'R$^2$')
    plt.xlabel('rank')
    plt.ylim([-2e-1,2e-1])
    
    plt.savefig('rank_vs_r2.png')
    return
    print(w)
    ypred = xts@w
    
    plt.plot(yts[:500],label = 'test')
    plt.plot(ypred[:500],label = 'pred')
    plt.legend()
    plt.savefig('demo_ts.png')
    plt.close()
    
    err = np.sum(np.square(ypred - yts),axis = 0)
    sc2 = np.sum(np.square(yts),axis = 0)
    r2 = 1 - err/sc2
    corr = np.einsum('ij,ij->j',ypred,yts)/sc2
    
    
    
    print(f'r2 = {r2}, corr = {corr}')
    
    
    plt.plot(ytr[:500],label = 'train')
    plt.plot(ypred[:500],label = 'pred')
    plt.legend()
    plt.savefig('demo_tr.png')
    plt.close()
    
    return
    
    Dataset.from_matrices()
    
    print(f'time_part = {time_part},y_index= {y_index},n_pop = {n_pop},n_iter = {n_iter},seed = {seed}')
    trdata = Dataset(time_part,train_flag=True,y_index=y_index)
    tstdata = Dataset(time_part,train_flag=False,y_index=y_index)
    tstdata.impose_pos_def()
    # tstdata.adopt_selection(trdata)
    gof = GeneticObjectiveFunc(trdata,tstdata,sparsity_limit=nz_lim)#sparsity_coeff=sparsity_coeff)
    np.random.seed(seed)
    n_bits =  trdata.XX.shape[0]
    r_cross = 0.9
    r_mut = 1/n_bits
    best,_,evals = genetic_algorithm(gof,n_bits,n_iter,n_pop,r_cross,r_mut)
    w_ = trdata.solve_system(best.nnz_pattern)
    r2tst = tstdata.get_rsqr_value(w_)
    r2trn = trdata.get_rsqr_value(w_)    
    kwargs['weights'] = w_
    kwargs['train_r2'] = r2trn
    kwargs['test_r2'] = r2tst
    kwargs['evals'] = evals
    
    

if __name__ == '__main__':
    main()