import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from selection_aware_bounds import WindowOracle
from test_path_bounds import enumerate_paths

class SelectionBoundsTests(unittest.TestCase):
    def test_exhaustive_conditioned_windows(self):
        rng=np.random.default_rng(20261110)
        for case in range(40):
            T=1+case%4;D=1+case%3
            p=rng.dirichlet([1,1,1],T)[:,:2]
            d=rng.dirichlet(np.ones(D),2).T;a=rng.dirichlet([1,1],2)
            if case==0:p[:]=.4;a[:]=1.
            paths=enumerate_paths(p,d,a);best=max(l for l,r,s in paths)
            eta=[0.,.01,.1,1.][case%4]
            feasible=[(l,r,s) for l,r,s in paths if l>=best-eta*T-1e-10]
            # A feasible threshold without using the tested oracle.
            threshold=float(np.median([r/T for l,r,s in feasible]))
            feasible=[(l,r,s) for l,r,s in feasible if r/T>=threshold-1e-12]
            mask=np.zeros(T);mask[case%T]=1.
            oracle=WindowOracle(p,d,a,mask,eta)
            for sign in [-1,1]:
                exact=max(sign*(p[np.arange(T),s]*mask).sum()/T for l,r,s in feasible)
                value=oracle.upper(sign,threshold,conditioned=True,max_queries=24)
                with self.subTest(case=case,sign=sign):self.assertGreaterEqual(value['upper'],exact-1e-8)

if __name__=='__main__':unittest.main(verbosity=2)
