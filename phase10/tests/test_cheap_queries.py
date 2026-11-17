import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from path_bounds import prepare,solve
from cheap_queries_and_winners import interval
from test_path_bounds import enumerate_paths

class CheapBoundTests(unittest.TestCase):
    def test_float32_extremal_means_are_accumulated_in_float64(self):
        for T in [7,127,999]:
            for value in [.1,.3,.65,.8]:
                p=np.full((T,2),value,dtype=np.float32)
                args=prepare(p,np.ones((1,2)),np.eye(2));base=solve(*args)
                lo,hi=interval(args,base,p,.005,[1.])
                exact=float(np.float64(p[0,0]))
                self.assertLessEqual(lo,exact);self.assertGreaterEqual(hi,exact)

    def test_fixed_queries_against_exhaustive_paths(self):
        rng=np.random.default_rng(20261111)
        for i in range(16):
            T=1+i%4;D=1+i%2
            p=rng.dirichlet([1,1,1],T)[:,:2].astype(np.float32)
            d=rng.dirichlet(np.ones(D),2).T.astype(np.float32);a=np.eye(2,dtype=np.float32)
            args=prepare(p,d,a);base=solve(*args);paths=enumerate_paths(p,d,a)
            for eta in [.001,.005,.01,.05]:
                feasible=[r/T for l,r,s in paths if l>=base[0]-eta*T-1e-9]
                for grid in [[1.],[.3,1.,3.],[.1,.3,1.,3.,10.,30.]]:
                    lo,hi=interval(args,base,p,eta,grid)
                    self.assertLessEqual(lo,min(feasible)+1e-9)
                    self.assertGreaterEqual(hi,max(feasible)-1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
