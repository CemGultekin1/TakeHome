import itertools
import sys
from typing import Tuple
import numpy as np
import os
from collections import defaultdict
from geneticalgorithm import geneticalgorithm as ga
from featsel.normaleqs import normal_eq_location,N_DAY_TIME,PROD_TYPES,N_CV
import dask.distributed
import dask
from featsel.constants import GENETIC_SOLUTIONS_FOLDER

def gen_sol_location(day_time_index:int,y_index:int,n_day_time :int = N_DAY_TIME)->str:
    """
    Returns the path to the genetic programming result. 
    Each model is trained on a particular portion of the day and for Y1 or Y2. 
        Args:
            "day_time_index"    : the particular partition of time out of "n_day_time" 
            "y_index"       : 0 or 1 for Y1 or Y2 predicting model
            "n_day_time"        : It has to be divisor of "N_DAY_TIME" global constant
        Returns:
            "path"          : The absolute path to the .npy file where the weights are stored
                    The file rests under the path specified in global variable "GENETIC_SOLUTIONS_FOLDER"
    """
    assert N_DAY_TIME % n_day_time  == 0
    
    folder = os.path.abspath(GENETIC_SOLUTIONS_FOLDER)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if n_day_time == N_DAY_TIME:
        filename= f't{day_time_index}_y{y_index}.npy'
    else:
        filename= f't{day_time_index}p{n_day_time}_y{y_index}.npy'
    return os.path.join(folder,filename)
    
class CostFunctor:
    """
    Cost function for genetic programming of feature selection.
    The cost function takes in a 0-1 pattern and an L2 regularizer coefficient.
    The 0-1 pattern determines which features to be used. 
    It fits a linear model using the normal equations and the regularizer.
    We use N_CV-fold cross-validation, separated across days.
    
    On each (train,test) dataset pair, solve the normal equations on train 
    and report its MSE on the test set. These MSE scores are averaged and 
    an R^2 score acquired.
    """
    def __init__(self,day_time_index:int = 0,n_day_time:int = N_DAY_TIME,y_index:int = 0,verbose :bool = True):
        """
        Args:
            "day_time_index"    : the particular partition of time out of "n_day_time"
            "n_day_time"        : must divisor of "N_DAY_TIME"
            "y_index"       : 0 or 1 depending on Y1 or Y2
            "verbose"       : prints whenever a new best is reached during training
        """
        assert day_time_index < N_DAY_TIME
        assert day_time_index >= 0
        assert y_index >= 0
        assert y_index <= 1
        self.verbose = verbose
        self.cv_normal_eqs = None
        self.reduc_inds = None 
        self.normaleqs = None
        self.best =dict(r2 = -np.inf,val = None)
        self.org_dim = None
        self.y_index = y_index
        self.day_time_index = day_time_index
        self.n_day_time = n_day_time
        if n_day_time != N_DAY_TIME:
            assert N_DAY_TIME % n_day_time == 0
        self.read_normal_equations()
        self.build_cross_validation_splits()   
    def get_full_solution(self,w:np.ndarray)->Tuple[float,np.ndarray]:
        """
        Solves the normal equations on the whole data.
        Returns the weights and the cross-validation score
        The weights are 0-filled and last entry is L2-regularization coefficient
            Args:
                "w"     :   0-1 vector specifying the selected features
                            appendedd by log10(l2-regularization-coefficient)
            Returns:
                "score"   : cross-validation score
                "weights" : the solutions to the normal equations
        """
        scr = -self.__call__(w)
        reg = np.power(10.,w[-1])
        log10reg = w[-1]
        b = w[:-1]>0.5
        nparts = len(self.cv_normal_eqs)
        xx,xy,yy = 0,0,0
        for i in range(nparts):
            xx_,xy_,yy_ = [self.cv_normal_eqs['test'][i][key] for key in PROD_TYPES]
            xx += xx_
            xy += xy_
            yy += yy_
        xx = xx[b,:]
        xx = xx[:,b]
        xy = xy[b,:]
        sc = np.amax(xx)
        w = np.linalg.solve(xx + np.eye(xx.shape[0])*reg*sc,xy[:,self.y_index])        
        w1 = np.zeros(len(b),dtype = np.float64)
        w1[b] =w
        if self.reduc_inds is not None:
            w = w1
            w1 = np.zeros(self.org_dim,dtype = np.float64)
            print(self.org_dim,len(self.reduc_inds),np.amax(self.reduc_inds),w.shape,flush = True)
            w1[self.reduc_inds] = w
        w1 = np.append(w1,log10reg)
        return scr,w1            
    def read_normal_equations(self,):
        """
            Reads the normal equation from files.
            These components are acquired from disjoint sets of trading days.
        """
        normaleqs = defaultdict(lambda : defaultdict(lambda : 0))
        for cvi,pt in itertools.product(range(N_CV),PROD_TYPES):
            ntotal = N_DAY_TIME//self.n_day_time
            tis = np.arange(self.day_time_index*ntotal,(self.day_time_index+1)*ntotal)
            for ti in tis:
                f = normal_eq_location(ti,cvi,pt)
                val = np.load(f)
                normaleqs[cvi][pt] += val
        self.normaleqs = normaleqs
        
        # the original dimension of the linear system (before reduction)
        self.org_dim = self.normaleqs[0]['xx'].shape[0]
    def build_cross_validation_splits(self,):
        """
            Converts the disjoint time sets into (train,test) pairs.
            For each pair, a particular time set is determined as test and 
            all else becomes train. 
        """
        cv_normal_eqs = {}
        keys = 'xx xy yy'.split()
        for i in range(len(self.normaleqs)):
            comps = defaultdict(lambda :0)
            for j in range(len(self.normaleqs)):
                if i == j:
                    continue
                for key in keys:
                    comps[key] += self.normaleqs[j][key]
            cv_normal_eqs[i] = comps
        self.cv_normal_eqs['train'] = cv_normal_eqs
        self.cv_normal_eqs['test'] = self.normaleqs
    def reduc(self,reltol :float = 1e-5):
        """
            Reduces the normal equations in dimensionality using QR-decomposition
            The features may be degenerate, i.e. linearly dependent.
            Such features appear as small residuals in R-matrix diagonals.
            Args:
                "reltol"     : lower limit for acceptable features
                        abs(R[i,i])/ABS_DIAG_R_MAX < reltol
        """
        if self.reduc_inds is not None:
            self.read_normal_equations()
        _,r = np.linalg.qr(self.normaleqs[0]['xx'])
        rdiag = np.abs(np.diag(r))
        rdiag = rdiag/np.amax(rdiag)
        inds, = np.where(rdiag > reltol)
        self.reduc_inds = inds
        for i in self.normaleqs:
            xx = self.normaleqs[i]['xx'] 
            xx = xx[inds,:]
            xx = xx[:,inds]
            self.normaleqs[i]['xx']  = xx
            xy = self.normaleqs[i]['xy'] 
            xy = xy[inds,:]
            self.normaleqs[i]['xy']  = xy
        self.build_cross_validation_splits()
    @property
    def ndim(self,):
        """
            Current dimensionality of the normal equations
            Note that this can change after a call to "reduc"
        """
        return self.cv_normal_eqs['train'][0]['xx'].shape[0]
    def compute_mse_sc2(self,b:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """
            Computes MSE score and average energy in the true Y1 or Y2 values(sc2)
            for each CV in numpy arrays
            Args:
                "b"   : 0-1 feature pattern and L2-reg coeff
            Returns:
                "mse"
                "sc2"
        """
        cv_sc2 = [] # total energy 
        cv_mse = []
        
        reg = np.power(10.,b[-1])
        b = b[:-1] > 0.5
        keys = 'xx xy yy'.split()
        nparts = len(self.cv_normal_eqs)
        
        for i in range(nparts):
            
            xx,xy,yy = [self.cv_normal_eqs['train'][i][key] for key in keys]
            sc= np.amax(xx)
            
            xx = xx[b,:]
            xx = xx[:,b]
            xy = xy[b,:]
            
            w = np.linalg.solve(xx/sc + reg*np.eye(xx.shape[0]),xy/sc)
            xx,xy,yy = [self.normaleqs['test'][i][key] for key in keys]
            xx = xx[b,:]
            xx = xx[:,b]
            xy = xy[b,:]
            
            err = w.T@xx@w - 2*w.T@xy + yy + np.linalg.norm(w)*1e-9
            err = np.diag(err)
            yy = np.diag(yy)
            
            try:
                assert(np.all(err>=0))
                assert(np.all(yy >=0))
            except:
                # raise Exception(
                #     f'np.linalg.norm(w) ={np.linalg.norm(w)},\t err = {err},\t yy = {yy}'
                # )
                err = np.inf*np.ones_like(err)
            cv_mse.append(err[self.y_index])
            cv_sc2.append(yy[self.y_index])
            
        return np.array(cv_mse),np.array(cv_sc2)
    def __call__(self,b:np.ndarray)->float:
        """
            Returns a cost score. This is negative of CV r-square value. 
            The genetic algorithm minimizes cost - maximizes r2.
            Args:
                "b" : feature selection pattern and a L2-regularization coeff
            Returns:
                "-r2": CV-averaged MSE scores turned into r-square value
                    -(1 - AVG_CV_MSE/SC2)
        """ 
        nnz = int(np.sum(b[:-1]))
        reg = b[-1]
        err,sc2 = self.compute_mse_sc2(b.copy())
        r2 =  1 - sum(err)/sum(sc2)
        if r2 > self.best['r2']:
            self.best['r2'] = r2
            self.best['val'] = b
            if self.verbose:
                formatter = "{:.3e}"
                regstr = formatter.format(b[-1])
                print(f't{self.day_time_index}y{self.y_index} = {formatter.format(r2)}, nnz = {nnz}, reg = {regstr}',flush = True)
        return -r2 + (nnz/375 + np.power(10.,reg+2))*1e-4

def greedy_sparsification(ftn:CostFunctor,b:np.ndarray):
    """
        Refines genetic programming results by through a greedy algorithm.
        It reduces the number of features and L2-regularization coefficient.
        Args:
            "ftn"   : The same CostFunctor used by the genetic programming
            "b"     : Feature selection pattern and L2-reg coeff
    """
    l2reg_decrement = 0.25
    r2 = -ftn(b)
    best = (b.copy(),r2)
    changed_flag = True
    while changed_flag:
        '''
            Try getting rid of one feature at a time 
            or reducing L2-regularization coefff
            until r2 value no longer improves.
        '''
        changed_flag = False        
        inds, = np.where(best[0][:-1])
        b = best[0]
        for i in inds:
            b[i] = False
            newr2= -ftn(b)
            if best[1] < newr2:
                best = (b.copy(),newr2)
                changed_flag = True
            if b[-1] > -12:
                b[-1] -= l2reg_decrement
                newr2= -ftn(b)
                if best[1] < newr2:
                    best = (b.copy(),newr2)
                    changed_flag = True
                b[-1] += l2reg_decrement
            b[i] = True    
    b,_ =best
    return b
    
    
@dask.delayed
def run_gen_alg(day_time_index:int,y_index:int,n_day_time:int = N_DAY_TIME):
    """
        A dask.delayed function that runs the genetic algorithm
        for one of Y1 or Y2 and specific time of the day.
        Args:
            "day_time_index"    : specifies time of the day out of "N_DAY_TIME" parts
            "y_index"       : 0 or 1 for Y1 or Y2
    """
    cost_fn = CostFunctor(day_time_index,n_day_time,y_index)
    cost_fn.reduc(reltol = 1e-5) # remove degenerate features
    
    # genetic algorithm parameters
    algparams = {'max_num_iteration': None,\
                'population_size':1000,\
                'mutation_probability':1/300,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'one_point',\
                'max_iteration_without_improv':200,}
    
    
    # genetic algorithm variable type
    # last value for l2-regularization coeff 
    # all else specifies the feature selection with 0-1    
    var_types = np.array([['int']]*cost_fn.ndim + [['real']])
    var_bound=np.array([[0,1]]*cost_fn.ndim + [[-12,-2]])
    model = ga(cost_fn,cost_fn.ndim+1,\
                variable_type_mixed = var_types,\
                progress_bar = False,\
                convergence_curve = False,\
                variable_boundaries = var_bound,\
                algorithm_parameters=algparams)
    
    model.run()
    
    # best feat selection
    solution=model.output_dict['variable']    
    
    # get the CV-score
    scr,_ = cost_fn.get_full_solution(solution)
    print(f'\ttime = {day_time_index}, y = {y_index}: nnz = {np.sum(np.abs(solution)>0).astype(int)}, scr = {scr}',flush = True)
    
    # sparsify the solution
    sparse_solution = greedy_sparsification(cost_fn,solution)

    # fills solution with zeros
    scr,weights = cost_fn.get_full_solution(sparse_solution)
    print(f'after sparsification:\n\t time = {day_time_index}, y = {y_index}: nnz = {np.sum(np.abs(weights)>0).astype(int)}, scr = {scr}',flush = True)
    address = gen_sol_location(day_time_index,y_index,n_day_time=n_day_time)
    np.save(address.replace('.npy',''),weights)
    return True
def main():
    ncpu = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_day_time = int(sys.argv[2])
    else:
        n_day_time = N_DAY_TIME
    
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)
    res = []
    for ti,yi in itertools.product(range(n_day_time),range(2)):
        res.append(run_gen_alg(ti,yi,n_day_time = n_day_time))
    client.compute(res,sync = True,scheduler='processes', num_workers=min(n_day_time*2,ncpu))
    
if __name__ == '__main__':
    main()