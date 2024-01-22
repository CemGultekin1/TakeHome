import itertools
from calibration.genetic import CostFunctor
import numpy as np
from calibration.normaleqs import read_parquet
import matplotlib.pyplot as plt
from calibration.test import LinearModel
        
def main():
    lm = LinearModel()
    # r2vals = []
    # for yi,ti in itertools.product(range(2),range(2)):
    #     cf = CostFunctor(ti,2,yi,False)
    #     w = lm.get_params(ti,yi)
    #     r2 = -cf(w,only_r2=True)
    #     r2str = "{:.3e}".format(r2)
    #     r2vals.append(r2str)
    # table = f"Y1|" + "|".join(r2vals[:2]) + "|\n"
    # table += f"Y2|" + "|".join(r2vals[2:]) + "|\n"
    # print(table)
    # return
    df = read_parquet()

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