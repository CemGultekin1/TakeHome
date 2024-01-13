
import matplotlib.pyplot as plt
import numpy as np
from data.feature import Categoricals
def main():
    ctgr = Categoricals()
    ctgr.collect_histograms()
    # save_histogram()
    return
    hists,chist = ctgr.read_histograms()
    edges =ctgr.edges
    entropy = -np.mean(hists*np.log(hists + 1e-19)/np.log(2),axis = 1)
    idx = np.argsort(chist[:,1])[::-1]
    hists = hists[idx,:]
    
    fig,axs = plt.subplots(4,1,figsize = (40,20))
    kwargs = dict(aspect = 1.,extent = (1,375,edges[0],edges[-1]))
    ax = axs[0]
    b = 8
    bhist = hists[:,:((hists.shape[1]//b)*b)].reshape([hists.shape[0],-1,b]).sum(axis = 2)
    pos = ax.imshow(bhist.T,**kwargs)    
    fig.colorbar(pos,ax = ax)

    ax  = axs[1]
    ax.semilogy(chist[idx])
    # ax.set_ylim([1e-4,1.])
    
    ax = axs[2]
    pos = ax.imshow(np.log10(bhist.T),**kwargs)
    fig.colorbar(pos,ax = ax)
    
    ax = axs[3]
    nstt = np.sum(hists > 1e-3,axis = 1)
    ax.plot(nstt)
    
    
    fig.savefig('hists.png')
  
    
if __name__ == '__main__':
    main()