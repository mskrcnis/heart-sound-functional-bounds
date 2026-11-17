"""Independent exhaustive graph oracle; does not use DP to compute references."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from path_bounds import FunctionalBounds, TINY, prepare, solve


def enumerate_paths(p,d,a):
    T,N=p.shape;D=len(d);all_paths=[]
    def visit(u,i,ll,rr,states):
        if u>=T:
            all_paths.append((ll,rr,np.array(states,dtype=int)))
            return
        for t in range(u+1,T+D):
            for duration in range(1,D+1):
                if max(0,min(t-duration,T-1))!=u:continue
                for s in range(N):
                    end=min(t,T)
                    edge=sum(np.log(float(p[k,s])+TINY) for k in range(u,end))
                    edge+=np.log(float(d[duration-1,s])+TINY)+np.log(float(a[i,s])+TINY)
                    reward=sum(float(p[k,s]) for k in range(u,end))
                    visit(t,s,ll+edge,rr+reward,states+[s]*(end-u))
    for i in range(N):visit(0,i,np.log(p[0,i]),0.,[])
    return all_paths


def witness_score(p,d,a,segments):
    T=len(p);seen=np.zeros(T,dtype=int);ll=rr=0.;last_start=None;last_previous=None
    for u,t,i,s,duration in segments:
        assert max(0,min(t-duration,T-1))==u
        if last_start is not None:assert t==last_start and s==last_previous
        end=min(t,T);seen[u:end]+=1
        ll+=sum(np.log(float(p[k,s])+TINY) for k in range(u,end))
        ll+=np.log(float(d[duration-1,s])+TINY)+np.log(float(a[i,s])+TINY)
        rr+=sum(float(p[k,s]) for k in range(u,end))
        last_start,last_previous=u,i
    assert last_start==0 and (seen==1).all()
    ll+=np.log(float(p[0,last_previous]))
    return ll,rr/T


class BoundTests(unittest.TestCase):
    def check_instance(self,p,d,a):
        paths=enumerate_paths(p,d,a)
        model=FunctionalBounds(p,d,a)
        ls=np.array([x[0] for x in paths]);cs=np.array([x[1]/len(p) for x in paths])
        self.assertAlmostEqual(model.optimum,ls.max(),places=10)
        exact=np.full(p.shape,-np.inf)
        for ll,rr,path in paths:
            for t,s in enumerate(path):exact[t,s]=max(exact[t,s],ll)
        np.testing.assert_allclose(model.max_marginal,exact,atol=1e-10,rtol=0.)
        for lam in [-4.,-.1,.1,4.]:
            got=solve(*prepare(p,d,a),lam=lam)
            self.assertAlmostEqual(got[0],max(ls+lam*cs*len(p)),places=10)
        for eps in [0.,.001,.2,1.,10.]:
            out=model.interval(eps,iterations=12)
            values=cs[ls>=ls.max()-eps-1e-10]
            self.assertLessEqual(out['lower'],values.min()+1e-9)
            self.assertGreaterEqual(out['upper'],values.max()-1e-9)
            self.assertLessEqual(out['lower'],out['feasible_lower']+1e-9)
            self.assertGreaterEqual(out['upper'],out['feasible_upper']-1e-9)
            for side in ['lower','upper']:
                ll,cc=witness_score(p,d,a,out['segments_'+side])
                self.assertAlmostEqual(ll,out['witness_'+side+'_log_score'],places=10)
                self.assertAlmostEqual(cc,out['feasible_'+side],places=10)
                self.assertGreaterEqual(ll,ls.max()-eps-1.1e-8)

    def test_random_exhaustive(self):
        rng=np.random.default_rng(20261108)
        for case in range(60):
            T=1+case%4;D=1+case%3;N=2
            p=rng.dirichlet(np.ones(N+1),T)[:,:N]
            d=rng.dirichlet(np.ones(D),N).T
            a=rng.dirichlet(np.ones(N),N)
            if case%2==0:a=np.eye(N)
            with self.subTest(case=case,T=T,D=D):self.check_instance(p,d,a)

    def test_exact_ties(self):
        self.check_instance(np.full((4,2),.4),np.full((2,2),.5),np.ones((2,2)))

    def test_likelihood_confidence_mismatch(self):
        p=np.array([[.56,.4],[.56,.4],[.4**4/.56**3*np.exp(-.0001),.4]])
        model=FunctionalBounds(p,np.ones((1,2)),np.eye(2))
        out=model.interval(.0002)
        self.assertAlmostEqual(model.base_confidence,.4,places=10)
        self.assertLess(out['feasible_lower'],.41)
        self.assertGreater(out['feasible_upper'],.41)

    def test_bad_input(self):
        with self.assertRaises(ValueError):prepare(np.array([[0.,.5]]),np.ones((1,2)),np.eye(2))

if __name__=='__main__':unittest.main(verbosity=2)
