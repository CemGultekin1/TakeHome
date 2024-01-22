import itertools
from featsel.constants import N_DAY_TIME
from featsel.genetic import gen_sol_location
import numpy as np
import pandas as pd
from featsel.normaleqs import read_parquet
import matplotlib.pyplot as plt
class LinearModel:
    max_time = 57600000
    min_time = 35101000
    def __init__(self,n_day_time:int = N_DAY_TIME//2):
        models = {(ti,yi) : gen_sol_location(ti,yi,n_day_time=n_day_time,makedirs_permit=False) for ti,yi in itertools.product(range(n_time),range(ny))}
        for key,path in models.items():
            weights_reg = np.load(path)
            models[key] = weights_reg[:-1]
        self.models = models
        self.n_day_time = n_day_time
    @staticmethod
    def _summarize(params):
        summary_str = []
        formatter = "{:.3e}"
        for i,w in enumerate(params):
            if np.abs(w) < 1e-9:
                continue
            w_str = formatter.format(w) 
            istr = str(i)       
            istr = (3-len(istr))*" " + istr
            w_str = (12- len(w_str))*" " + w_str
            summary_str.append(f'X({istr}) : {w_str}')
        return summary_str
    def summarize(self,):
        summ = ""
        for key,prm in self.models.items():
            nnz = np.sum(np.abs(prm)>0)
            summ += f"T({key[0]}/{self.n_time}),Y{key[1]+1}: #feats = {nnz}\n\t"
            sstr = LinearModel._summarize(prm)
            summ += ",\n\t".join(sstr) + "\n"
        print(summ)
    def __call__(self,x:pd.Series):
        t = x['time']        
        reltime = (t- self.min_time)/(self.max_time - self.min_time)
        ti = int(np.floor(reltime*self.n_time))
        ti = np.maximum(np.minimum(ti,self.n_time - 1),0)
        w0 = self.models[(ti,0)]
        w1 = self.models[(ti,1)]
        qs = np.array([x[f'Q{i+1}'] for i in range(2)])
        qs = qs > 0.99999
        feats = np.array([x[f'X{i+1}'] for i in range(375)])
        mask = np.isnan(feats) | (np.abs(feats)>999)
        feats[mask] = 0
        ys = dict(
            Y1pred = w0 @ feats if qs[0] else np.nan,
            Y2pred = w1 @ feats if qs[1] else np.nan
        )
        return pd.Series(data = ys, index = list(ys.keys()))
        
def main():
    lm = LinearModel()
    df = read_parquet()
    lm.summarize()
    return
    dfpart = df.partitions[100]
    # dfpart = dfpart[dfpart['time'] < dfpart['time'].max()/2]
    ypred = dfpart.apply(lm,axis = 1).compute()
    ytrue = dfpart[['Y1','Y2']]
    ypredval = ypred.values
    ytrueval = ytrue.values.compute()
            
    ts = np.array_split(np.arange(22500),20)
    for ti,tsl in enumerate(ts):
        fig,axs = plt.subplots(2,1,figsize = (10,5))
        axs = axs.flatten()
        for i,ax in enumerate(axs):
            ypv = ypredval[tsl,i]
            ypt = ytrueval[tsl,i]
            ax.plot(tsl/4,ypv,label = f'Y{i+1}-pred')
            ax.plot(tsl/4,ypt,label = f'Y{i+1}-true')
            ax.legend()
        fig.savefig(f'demo-{ti}.png')
        plt.close()
    


if __name__ == '__main__':
    main()