import itertools
from featsel.genetic import FitnessFunctor,gen_sol_location
import numpy as np

from featsel.normaleqs import N_TIME
def greedy_sparsify(ti,yi):
    ftn = FitnessFunctor(ti,yi,verbose=False)
    loc= gen_sol_location(ti,yi)
    print(f'loaded weights = {loc}')
    w = np.load(loc)
    print(f'w.shape = {w.shape}')
    if len(w) < 376:
        w = np.append(w,-np.inf)
    b = w.copy()
    
    reginc = 0.5
    
    b[:-1] = np.abs(w[:-1]) != 0
    r2 = -ftn(b)
    best = (b.copy(),r2)
    changed_flag = True
    while changed_flag:
        print(f'nnz = {np.sum(best[0][:-1]).astype(int)}, r2 = {best[1]}, reg = {best[0][-1]}\n\n')
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
                b[-1] -= reginc
                newr2= -ftn(b)
                if best[1] < newr2:
                    best = (b.copy(),newr2)
                    changed_flag = True
                b[-1] += reginc
            b[i] = True    
    b,_ =best
    scr,weights = ftn.get_full_solution(b)
    print(f'final scr = {scr}')
    np.save(loc.replace('.npy',''),weights)

def main():
    for ti,yi in itertools.product(range(N_TIME),range(2)):
        greedy_sparsify(ti,yi)

if __name__== '__main__':
    main()