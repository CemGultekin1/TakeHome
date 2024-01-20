import itertools
from learning.genetic import FitnessFunctor,gen_sol_location
import numpy as np

from learning.prods import N_TIME
def greedy_sparsify(ti,yi):
    ftn = FitnessFunctor(ti,yi,verbose=False)
    loc= gen_sol_location(ti,yi)
    print(f'loaded weights = {loc}')
    w = np.load(loc)
    b = np.abs(w) != 0
    r2 = -ftn(b)
    best = (b.copy(),r2)
    changed_flag = True
    while changed_flag:
        print(f'nnz = {np.sum(best[0])}, r2 = {best[1]}')
        changed_flag = False        
        inds, = np.where(best[0])     
        b = best[0]
        for i in inds:
            b[i] = False
            newr2= -ftn(b)
            if best[1] < newr2:
                best = (b.copy(),newr2)
                changed_flag = True
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