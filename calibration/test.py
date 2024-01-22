import itertools
from calibration.constants import CHOSEN_N_DAY_TIME
from calibration.genetic import gen_sol_location
import numpy as np
import pandas as pd

class LinearModel:
    """
    Loads the calibration results 
    Evaluates it on a given pandas.Series
    """
    max_time = 57600000
    min_time = 35101000
    def __init__(self,n_day_time:int = CHOSEN_N_DAY_TIME):
        addresses = {(ti,yi) : gen_sol_location(ti,yi,n_day_time=n_day_time,makedirs_permit=False) \
            for ti,yi in itertools.product(range(n_day_time),range(2))}
        weights = {}
        log10l2regs = {}
        for key,path in addresses.items():
            weights_reg = np.load(path)
            weights[key] = weights_reg[:-1]
            log10l2regs[key] = weights_reg[-1]
        self.weights = weights
        self.l2regs = log10l2regs
        self.n_day_time = n_day_time
    def get_params(self,i,j):
        w = self.weights[(i,j)]
        w = np.abs(w) > 1e-9
        reg = self.l2regs[(i,j)]
        return np.append(w,reg)
    def __call__(self,x:pd.Series):
        """
        Transforms dataframe to output predicted Y1 and Y2
        It needs Q and X column values
        """
        t = x['time']        
        reltime = (t- self.min_time)/(self.max_time - self.min_time)
        ti = int(np.floor(reltime*self.n_day_time))
        ti = np.maximum(np.minimum(ti,self.n_day_time - 1),0)
        w0 = self.weights[(ti,0)]
        w1 = self.weights[(ti,1)]
        qs = np.array([x[f'Q{i+1}'] for i in range(2)])
        qs = qs > 0.99999
        feats = np.array([x[f'X{i+1}'] for i in range(375)])
        mask = np.isnan(feats) | (np.abs(feats)>999)
        feats[mask] = 0
        ys = dict(
            Y1 = w0 @ feats if qs[0] else np.nan,
            Y2 = w1 @ feats if qs[1] else np.nan
        )
        return pd.Series(data = ys, index = list(ys.keys()))
        