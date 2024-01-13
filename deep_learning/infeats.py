import itertools
import os
import numpy as np
from file_utils import MixedTypeStorage
from genetic import GENETIC_FOLDER
import matplotlib.pyplot as plt
from data.feature import CategoricalFeatureTransform,Categoricals
def find_best_feature_sel(time_part,y_index,max_feat_sel = np.inf):
    fldrs = [os.path.join(GENETIC_FOLDER,f) for f in os.listdir(GENETIC_FOLDER) ]
    fldrs = [f for f in fldrs if os.path.isdir(f)]    
    ys = []
    for i,path in enumerate(fldrs):
        solsdict = MixedTypeStorage()
        solsdict.load_from_file(path)
        if tuple(solsdict['time_part']) != time_part:
            continue
        nz = solsdict['nnz']
        r2 = solsdict['r2_values'].mean()
        if y_index != solsdict['y_index']:
            continue
        ys.append([nz,r2])
    nzs,r2s = tuple(zip(*ys))
    r2s = np.array(r2s)
    nzs = np.stack(nzs,axis = 0)
    nnz = nzs.sum(axis = 1)
    mask = nnz <= max_feat_sel
    r2s = r2s[mask]
    nzs = nzs[mask,:]
    i = np.argmax(r2s)
    return r2s[i],nzs[i]

class SelectedFeatures(CategoricalFeatureTransform):
    def __init__(self, cft:CategoricalFeatureTransform) -> None:
        self.__dict__.update(cft.__dict__)
        self.subfeature_set = {}
        self.padded_feature_len = -1
    def gather_indices(self,):
        for ti, yi in itertools.product(range(4),range(2)):
            _,nzp = find_best_feature_sel((ti,4),yi)
            self.subfeature_set[(ti,yi)] = nzp
        n = [0,0]
        bndrs = [[],[]]
        for ti in range(4):
            n0,n1 = n
            n[0] += self.subfeature_set[(ti,0)].sum()
            n[1] += self.subfeature_set[(ti,1)].sum()
            n0_,n1_ = n
            bndrs[0].append((n0,n0_))
            bndrs[1].append((n1,n1_))
        self.bndrs = bndrs
        self.padded_feature_len = max(n[0],n[1])
    @property
    def num_features(self,):                     
        return self.padded_feature_len
    def __call__(self, x: np.ndarray,ti:int):
        y =  super().__call__(x/self.stds)
        y0 = np.zeros(self.padded_feature_len)
        y1 = np.zeros(self.padded_feature_len)
        y0[self.bndrs[0][ti][0]:self.bndrs[0][ti][1]] = y[self.subfeature_set[(ti,0)]]
        y1[self.bndrs[1][ti][0]:self.bndrs[1][ti][1]] = y[self.subfeature_set[(ti,1)]]
        return np.concatenate([y0,y1],axis = 0)

def main():
    sf = SelectedFeatures(Categoricals().determine_ctgr())
    sf.gather_indices()
    union_val = np.stack(list(sf.subfeature_set.values()),axis = 0)
    fig,axs = plt.subplots(union_val.shape[0],1)
    axs = axs.flatten()
    for ax,uval in zip(axs,union_val):
        ax.plot(uval)
    plt.savefig('union.png')
    print(sf.padded_feature_len)
if __name__ == '__main__':
    main()