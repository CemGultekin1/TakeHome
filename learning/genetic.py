import itertools
import numpy as np
import os
from collections import defaultdict
from geneticalgorithm import geneticalgorithm as ga
from learning.prods import prod_location,N_TIME,PROD_TYPES,N_CV
import dask.distributed
import dask

SOLUTIONS_FOLDER = 'genetic_solutions'
def gen_sol_location(time_index,y_index):
    folder = os.path.abspath(SOLUTIONS_FOLDER)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename= f't{time_index}_y{y_index}.npy'
    return os.path.join(folder,filename)
    
class FitnessFunctor:
    def __init__(self,ti:int = 0,yi:int = 0,verbose = True):
        assert ti < N_TIME
        assert ti >= 0
        assert yi >= 0
        assert yi <= 1
        self.verbose = verbose
        self.cvcomps = None
        self.reduc_inds = None 
        self.lincomps = None
        self.best =dict(r2 = -np.inf,val = None)
        self.org_dim = None
        self.yi = yi
        self.ti = ti
        self.read_from_folder(ti,yi)
        self.collect_cv_comps()   
    def get_full_solution(self,w):
        scr = -self.__call__(w)
        reg = np.power(10.,w[-1])
        b = w[:-1]>0.5
        keys = 'xx xy yy'.split()
        nparts = len(self.cvcomps)
        xx,xy,yy = 0,0,0
        for i in range(nparts):
            xx_,xy_,yy_ = [self.cvcomps[i][key] for key in keys]
            xx += xx_
            xy += xy_
            yy += yy_
        xx = xx[b,:]
        xx = xx[:,b]
        xy = xy[b,:]
        w = np.linalg.solve(xx + np.eye(xx.shape[0])*reg,xy[:,self.yi])        
        ww = np.zeros(len(b),dtype = np.float64)
        ww[b] =w
        if self.reduc_inds is not None:
            w = ww
            ww = np.zeros(self.org_dim,dtype = np.float64)
            print(self.org_dim,len(self.reduc_inds),np.amax(self.reduc_inds),w.shape)
            ww[self.reduc_inds] = w
        return scr,ww            
    def read_from_folder(self,ti,yi):
        lincomps = defaultdict(lambda : {})
        for cvi,pt in itertools.product(range(N_CV),PROD_TYPES):
            f = prod_location(ti,cvi,pt)
            lincomps[cvi][pt] = np.load(f)
        self.lincomps = lincomps
        self.org_dim = self.lincomps[0]['xx'].shape[0]
    def collect_cv_comps(self,):
        cvcomps = {}
        keys = 'xx xy yy'.split()
        for i in range(len(self.lincomps)):
            comps = defaultdict(lambda :0)
            for j in range(len(self.lincomps)):
                if i == j:
                    continue
                for key in keys:
                    comps[key] += self.lincomps[j][key]
            cvcomps[i] = comps
        self.cvcomps = cvcomps
    def reduc(self,tol = 1e-5):
        if self.reduc_inds is not None:
            self.read_from_folder()
        _,r = np.linalg.qr(self.lincomps[0]['xx'])
        rdiag = np.abs(np.diag(r))
        rdiag = rdiag/np.amax(rdiag)
        inds, = np.where(rdiag > tol)
        self.reduc_inds = inds
        print(f'from {self.org_dim} to {len(inds)} many dimensions')
        for i in self.lincomps:
            xx = self.lincomps[i]['xx'] 
            xx = xx[inds,:]
            xx = xx[:,inds]
            self.lincomps[i]['xx']  = xx
            xy = self.lincomps[i]['xy'] 
            xy = xy[inds,:]
            self.lincomps[i]['xy']  = xy
        self.collect_cv_comps()
    @property
    def ndim(self,):
        return self.cvcomps[0]['xx'].shape[0]
    def get_err_sc2(self,b):
        yysum = []
        errsum = []
        reg = np.power(10.,b[-1])
        b = b[:-1] > 0.5
        keys = 'xx xy yy'.split()
        nparts = len(self.cvcomps)
        for i in range(nparts):
            xx,xy,yy = [self.cvcomps[i][key] for key in keys]
            xx = xx[b,:]
            xx = xx[:,b]
            xy = xy[b,:]
            sc= np.amax(xx)
            w = np.linalg.solve(xx/sc + reg*np.eye(xx.shape[0]),xy/sc)
            xx,xy,yy = [self.lincomps[i][key] for key in keys]
            xx = xx[b,:]
            xx = xx[:,b]
            xy = xy[b,:]
            err = w.T@xx@w - 2*w.T@xy + yy #+ np.linalg.norm(w)*1e-9
            err = np.diag(err)
            yy = np.diag(yy)
            try:
                assert(np.all(err>=0))
                assert(np.all(yy >=0))
            except:
                raise Exception(
                    f'np.linalg.norm(w) ={np.linalg.norm(w)},\t err = {err},\t yy = {yy}'
                )
                # err = np.inf*np.ones_like(err)
            errsum.append(err[self.yi])
            yysum.append(yy[self.yi])
        return np.array(errsum),np.array(yysum)
    def __call__(self,b):        
        err,sc2 = self.get_err_sc2(b.copy())
        r2 =  1 - sum(err)/sum(sc2)
        if r2 > self.best['r2']:
            self.best['r2'] = r2
            self.best['val'] = b
            if self.verbose:
                formatter = "{:.3e}"
                regstr = formatter.format(b[-1])
                nnz = int(np.sum(b[:-1]))
                print(f't{self.ti}y{self.yi} = {formatter.format(r2)}, nnz = {nnz}, reg = {regstr}')
        return -r2

@dask.delayed
def run_gen_alg(ti,yi):
    fitness_fn = FitnessFunctor(ti,yi)
    fitness_fn.reduc(tol = 1e-5)
    algparams = {'max_num_iteration': None,\
                'population_size':1000,\
                'mutation_probability':1/300,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'one_point',\
                'max_iteration_without_improv':200,}
    var_types = np.array([['int']]*fitness_fn.ndim + [['real']])
    var_bound=np.array([[0,1]]*fitness_fn.ndim + [[-12,-1]])
    model = ga(fitness_fn,fitness_fn.ndim+1,\
                variable_type_mixed = var_types,\
                progress_bar = False,\
                convergence_curve = False,\
                variable_boundaries = var_bound,\
                algorithm_parameters=algparams)
    model.run()
    solution=model.output_dict
    scr,weights = fitness_fn.get_full_solution(solution['variable'])
    address = gen_sol_location(ti,yi)
    print(f'time = {ti}, y = {yi}, scr = {scr}:\t address = {address}')
    np.save(address.replace('.npy',''),weights)
    return True
def main():
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)
    res = []
    for ti,yi in itertools.product(range(N_TIME),range(2)):
        res.append(run_gen_alg(ti,yi))
    client.compute(res,sync = True)
    
if __name__ == '__main__':
    main()