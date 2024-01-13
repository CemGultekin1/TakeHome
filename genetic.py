import itertools
from typing import List, Tuple
import dask.dataframe as dataframe
import os
import numpy as np
from numpy.random import randint,rand
import matplotlib.pyplot as plt
from data.statistics import get_values,get_statistics_path
from data.feature import Categoricals
from data.normalization import Normalization
import dask
from distributed import Client
from data.statistics import FEATURE_FOLDER_POINTER
from sklearn.linear_model import Lasso
from file_utils import MixedTypeStorage, NumpyArrayDictionary
GENETIC_FOLDER = os.path.join(FEATURE_FOLDER_POINTER,'genetic')

class Dataset:
    def __init__(self,xx,xy,yy,y_index:int = 0):
        self.XX,self.XY,self.YY = xx,xy[:,y_index],yy[y_index,y_index]
        norm = Normalization(n_type='vector',feat_transform=Categoricals().determine_ctgr())
        xydict = norm.collect_normalizations_from_file()
        xn = np.linalg.inv(xydict['x'])
        self.XX = xn.T@self.XX@xn
        self.XY = xn.T@self.XY
        
    def impose_pos_def(self):
        eigv,_ = np.linalg.eig(self.XX)
        mineig = np.amin(np.real(eigv))
        print(f'impose_pos_def : min-eig = {mineig}')
        if mineig < 0:
            self.XX += np.eye(self.XX.shape[0])*(-mineig)                

    def get_subsystem(self,c = None):
        if c is None:
            c = np.ones(self.XX.shape[0],dtype = bool)
        xx = self.XX[c,:]
        xx = xx[:,c]
        xy = self.XY[c]
        return xx,xy
    def solve_system(self,c = None,reg_const = None):
        xx,xy = self.get_subsystem(c)
        if reg_const is None:
            reg_const = 0.
        try:
            w_ = np.linalg.solve(xx,xy)
        except:
            w_ = np.linalg.solve(xx + 1e-9*np.eye(xx.shape[0]),xy)
        w = np.zeros(self.XX.shape[0])
        w[c] = w_
        return w
    def lasso_fit(self,alpha:float):
        xx,xy = self.XX,self.XY
        scl = np.amax(xx)
        mdl = Lasso(alpha = alpha)
        mdl.fit(xx/scl,xy/scl)
        return mdl.coef_.reshape([-1,])
    def get_rsqr_value(self,w):
        if w is None:
            return np.inf
        err = w.T@(self.XX@w) - 2*(w.T@self.XY) + self.YY + np.linalg.norm(w)**2*1e-7
        if np.any(np.isnan(w)) or np.isnan(err):
            return np.inf
        assert err >= 0
        r2 = 1 - err/self.YY
        return r2.squeeze()
    def constrain_features(self, flag:np.ndarray):
        self.XX = self.XX[flag,:]
        self.XX = self.XX[:,flag]
        self.XY = self.XY[flag,]
class CVDataset(Dataset):
    def __init__(self,time_partition:Tuple[int,int],n_part :int = None,y_index = 0) -> None:
        x = get_statistics_path(time_partition)
        for key,path in x.items():
            x[key] = np.load(path).astype(np.double)
        
        for key in x:
            numb = x[key].shape[0]
            break
        self.cross_pairs= []
        n_part = min(numb,n_part)
        parts = np.array_split(np.arange(numb),n_part,)
        print(f'parts = {parts}')
        assert len(parts) == n_part
        flags = []
        for part in parts:
            flg = np.ones(numb,dtype = bool)
            flg[part] = 0
            flags.append(flg)
        for i in range(n_part):
            flg = flags[i]
            dataset_args1 = []
            dataset_args2 = []
            for key in x:
                dataset_args1.append(np.sum(x[key][flg],axis = 0))
                dataset_args2.append(np.sum(x[key][~flg],axis = 0))   
            trdata = Dataset(*dataset_args1,y_index = y_index)
            tsdata = Dataset(*dataset_args2,y_index = y_index)
            self.cross_pairs.append((trdata,tsdata))
        args = []
        for key in x:
            args.append(np.sum(x[key],axis = 0))
        super().__init__(*args)
    def constrain_all_features(self, flags):
        self.constrain_features(flags)
        for tr,ts in self.cross_pairs:
            tr.constrain_features(flags)
            ts.constrain_features(flags)
    def find_lasso_fits(self,alpha:float):
        nnz_patterns = []
        r2s = []
        for tr,ts in self.cross_pairs:
            w = tr.lasso_fit(alpha)
            nnz_pattern = np.abs(w)>1e-9
            nnz_patterns.append(nnz_pattern)
            w = tr.solve_system(nnz_pattern)
            r2s.append(ts.get_rsqr_value(w))
        return np.stack(nnz_patterns,axis = 0),np.array(r2s)
    
    def get_rsqure_values(self,c):
        r2s = []
        for tr,ts in self.cross_pairs:
            w = tr.solve_system(c)
            r2s.append(ts.get_rsqr_value(w))
        return np.array(r2s)
    def __call__(self,c):
        r2s = self.get_rsqure_values(c)
        return np.mean(r2s)        

class Chromosome:
    def __init__(self,n_bits :int=None,nnz_pattern:np.ndarray = None,init_nnz :int= None):
        if nnz_pattern is None:
            if init_nnz is None:
                nnz_pattern = randint(0,2,n_bits) == 1
            else:
                nnz_pattern = np.zeros(n_bits,dtype = bool)
                nnz_pts = np.random.choice(n_bits,init_nnz,replace = False)
                nnz_pattern[nnz_pts] = True
        self.nnz_pattern = nnz_pattern
        self.n_bits = len(self.nnz_pattern)
    def copy(self,):
        x = Chromosome(nnz_pattern=self.nnz_pattern.copy())
        return x
    def crossover(self,chr:'Chromosome',r_cross):
         # check for recombination
        if rand() > r_cross:        
            return self, chr
        pt = randint(1, self.n_bits-2)
        c1 = np.concatenate([self.nnz_pattern[:pt],chr.nnz_pattern[pt:]])
        c2 = np.concatenate([chr.nnz_pattern[:pt],self.nnz_pattern[pt:]])
        chr1 = Chromosome(self.n_bits,c1)
        chr2 = Chromosome(self.n_bits,c2)
        return chr1,chr2
    def mutate_(self,i):
        self.nnz_pattern[i] = ~self.nnz_pattern[i]
    @property
    def nnz(self,):
        return np.sum(self.nnz_pattern)
    def mutation(self,r_mut):
        for i in range(self.n_bits):
            # check for a mutation
            if rand() < r_mut:
                # flip the bit
                self.mutate_(i)
    def is_all_zero(self,):
        return np.all(~self.nnz_pattern)
    def to_file(self,iter_num:iter,folder_name:str):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        path = os.path.join(folder_name,f'best_{iter_num}')
        np.save(path,self.nnz_pattern)
# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
      

class GeneticObjectiveFunc:
    def __init__(self,cv_evaluator:CVDataset,sparsity_limit :int= 25) -> None:
        self.cv_evaluator = cv_evaluator
        self.sparsity_limit = sparsity_limit
    def __call__(self, chr:Chromosome):
        if chr.is_all_zero():
            return np.inf
        r2 = self.cv_evaluator(chr.nnz_pattern)
        sppen = (chr.nnz - self.sparsity_limit)/chr.n_bits
        return -r2 + np.maximum(0,sppen)



    
# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut,seed_pop:List[Chromosome] = []):
    # initial population of random bitstring
    pop = [Chromosome(n_bits=n_bits) for _ in range(n_pop - len(seed_pop))] + seed_pop
    pop[0] = Chromosome(nnz_pattern = np.ones(n_bits,dtype = bool))
    # keep track of best solution
    best, best_eval = pop[0], objective(pop[0])
    best_evals = []
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i].copy(), scores[i]
                assert np.abs(objective(best) - best_eval) < 1e-12
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop-1, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in p1.crossover(p2, r_cross):
                # mutation
                c.mutation(r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        best_evals.append(best_eval)  
        print(f'#{gen}: {best_eval}, nnz = {best.nnz}')        
    return best, best_eval,np.array(best_evals)

def get_dataset():
    parquet_directory = 'qr_takehome'
    parquet_files = [os.path.join(parquet_directory, f) for f in os.listdir(parquet_directory) if f.endswith('.parquet')]
    return dataframe.read_parquet(parquet_files)

@dask.delayed
def run_sparse_genetic_alg(**kwargs):
    time_part = kwargs.get('time_part',(0,4))
    n_pop = kwargs.get('n_pop',2**10)
    n_iter = kwargs.get('n_iter',2**8)
    seed = kwargs.get('seed',0)
    filename = kwargs.pop('filename','temp')
    y_index = kwargs.get('y_index',0)
    nz_lim = kwargs.get('nz_lim',20)
    seed_pop = kwargs.get('seed_pop',[])
    r_mut = kwargs.get('r_mut',None)
    r_cross = kwargs.get('r_cross',0.9)
    np_seed_pop = np.stack(seed_pop,axis = 0)
    kwargs['seed_pop'] = np_seed_pop
    seed_pop = [Chromosome(nnz_pattern= f) for f in seed_pop]
    mts = MixedTypeStorage.from_dict(kwargs)
    print(f'time_part = {time_part},y_index= {y_index},n_pop = {n_pop},n_iter = {n_iter},seed = {seed}, nz_lim = {nz_lim}')
    trdata = CVDataset(time_part,y_index=y_index,n_part=5)
    
    gof = GeneticObjectiveFunc(trdata,sparsity_limit=nz_lim)
    
    
    np.random.seed(seed)
    n_bits =  trdata.XX.shape[0]
    r_mut = 2/n_bits if r_mut is None else r_mut
    
    
    best,best_eval,evals = genetic_algorithm(gof,n_bits,n_iter,n_pop,r_cross,r_mut,seed_pop = seed_pop)
    # w_ = trdata.solve_system(best.nnz_pattern)
    # mts['weights'] = w_
    mts['r2_values'] = trdata.get_rsqure_values(best.nnz_pattern)
    print(f'mts[r2_values].mean(),best_eval = {mts["r2_values"].mean(),-best_eval}')
    mts['evals'] = evals
    mts['nnz'] = best.nnz_pattern
    mts.save_to_file(filename)
    return

class LassoFeatureFinder:
    def __init__(self,) -> None:        
        self.seed_pops = {}
    def is_explored(self,time_part,y_index):
        key = time_part[0],y_index
        return key in self.seed_pops,key
    def find_features(self,time_part,y_index):
        flag,key = self.is_explored(time_part,y_index)
        if flag:
            return self.seed_pops[key]        
        trdata = CVDataset(time_part,y_index=y_index,n_part=5)
        alphas = np.power(10.,np.arange(-12,-4))
        nnz_pattern_seeds = []
        lasso_results = []
        for alpha in alphas:
            nnz,r2s = trdata.find_lasso_fits(alpha)
            nnz_pattern_seeds.extend(np.array_split(nnz,nnz.shape[0],axis = 0))
            lasso_results.append((alpha,nnz,r2s))
        for alpha,nnz,r2s in lasso_results:
            print(alpha, np.sum(nnz,axis = 1),r2s.tolist())
        seed_pop = []
        for nnzp in nnz_pattern_seeds:
            seed_pop.append(nnzp.flatten())
        self.seed_pops[key] = seed_pop
        return seed_pop
    
def train_genalgs():
    nz_lims = [16,32,64,128,256]
    rslts = []
    
    if not os.path.exists(GENETIC_FOLDER):
        os.mkdir(GENETIC_FOLDER)
    lassff = LassoFeatureFinder()
    T = 4
    for time_part,y_index,nz in itertools.product(range(T),range(2),nz_lims):
        time_part = (time_part,T)
        seed_pop = lassff.find_features(time_part,y_index)
        filename = f't{time_part[0]}_y{y_index}_nz{nz}'
        path = os.path.join(GENETIC_FOLDER,filename)
        rslts.append(run_sparse_genetic_alg(filename = path,\
            time_part = time_part,nz_lim = nz,\
                n_iter = 2**9,n_pop = 2**11,y_index= y_index,seed_pop = seed_pop))
        rslts.append(filename)
    Client().compute(rslts,sync = True)
def reiterate():
    fldrs = [os.path.join(GENETIC_FOLDER,f) for f in os.listdir(GENETIC_FOLDER) ]
    fldrs = [f for f in fldrs if os.path.isdir(f)]    
    gen_results = {}
    for path in fldrs:
        solsdict = MixedTypeStorage()
        solsdict.load_from_file(path)
        key = (tuple(solsdict['time_part']),solsdict['y_index'])
        nnz = solsdict['nnz'] 
        if key not in gen_results:
            gen_results[key] = []
        gen_results[key].append(nnz)
    rslts= []
    for ((time_part,y_index),nnzs),seed in itertools.product(gen_results.items(),range(4)):
        print(f'time_part,y_index = {time_part,y_index}, nnz vecs = #{len(nnzs)}')
        filename = f't{time_part[0]}_y{y_index}_merged'
        path = os.path.join(GENETIC_FOLDER,filename)
        rslts.append(run_sparse_genetic_alg(filename = path,\
            time_part = time_part,nz_lim = nnzs[0].shape[0],\
                n_iter = 2**9,n_pop = 2**11,y_index= y_index,\
                        seed_pop = nnzs,r_mut = 0.1,seed = seed))
    Client().compute(rslts,sync = True)
def main():    
    train_genalgs()
    reiterate()
    for tp in range(4):
        plot_summary(time_part = (tp,4))
    return

def plot_summary(time_part = (0,4),):
    fldrs = [os.path.join(GENETIC_FOLDER,f) for f in os.listdir(GENETIC_FOLDER) ]
    fldrs = [f for f in fldrs if os.path.isdir(f)]    
    ys = []
    for i,path in enumerate(fldrs):
        solsdict = MixedTypeStorage()
        solsdict.load_from_file(path)
        if tuple(solsdict['time_part']) != time_part:
            continue
        nz = np.sum(solsdict['nnz'])
        r2 = solsdict['r2_values'].mean()
        yindex = solsdict['y_index']
        ys.append([yindex,nz,r2])
        print(path,yindex,nz,r2)
    yinds,nzs,r2s = tuple(zip(*ys))
    nzs = np.array(nzs)
    r2s = np.array(r2s)
    
    fig,axs = plt.subplots(1,1,figsize = (5,5))
    yinds = np.array(yinds)
    ymask = [yinds==i for i in range(2)]
    markers = ['x','+']
    ax = axs
    for i in range(2):
        ax.plot(nzs[ymask[i]],r2s[ymask[i]],marker = markers[i],linestyle = 'None')
    ax.set_title('Cross-Validation R$^2$')
    ax.set_xlabel('# num features')
    fig.savefig(f'summary_{time_part[0]}_{time_part[1]}.png')
    
if __name__ == '__main__':
    main()