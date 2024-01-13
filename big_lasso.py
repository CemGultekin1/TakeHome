from sklearn.linear_model import Lasso
import os
import numpy as np
def main():
    root = 'features/base_case/inner_prods'
    files = {f.split('_')[-1].split('.npy')[0]:os.path.join(root,f) for f in os.listdir(root) if '.npy'in f}
    print(files)
    rslts = {}
    for key,path in files.items():
        rslts[key] = np.load(path)
    alpha = 1e-5
    
    
    def tr_tst_part(i,xx):
        xxtr = np.sum(xx[:i],axis = 0) + np.sum(xx[i+1:],axis = 0)
        xxts = xx[i]
        return xxtr,xxts
    r2s = []
    for i in range(rslts['xx'].shape[0]):   
        xxtr,xxts = tr_tst_part(i,rslts['xx'])
        xytr,xyts = tr_tst_part(i,rslts['xy'][...,0])
        _,yyts = tr_tst_part(i,rslts['yy'][...,0,0])
        mdl = Lasso(alpha= alpha,max_iter=500)
        mdl.fit(xxtr/np.amax(xxtr),xytr/np.amax(xxtr))
        w = mdl.coef_.flatten()
        mse = (xxts@w).T@w - 2* xyts.T@w + yyts**2
        sc2 = yyts**2
        r2 = 1 - mse/sc2
        print(r2)
        r2s.append(r2)
    print(r2s)
        

if __name__ == '__main__':
    main()