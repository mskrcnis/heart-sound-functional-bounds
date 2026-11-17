# Public release adaptation notice, 2026-11-17: license notice added;
# numerical implementation unchanged. Derived boundary semantics: University
# of Cambridge, 2022. GAP Available Source License v1.0; see LICENSE.
"""Exploratory selection-conditioned additive functional bounds.

Standard nonnegative Lagrange multipliers and a cutting-plane master.
Legacy HSMM graph adapter retains GAP ASL v1.0, see phase9/HSMM_LICENSE.md.
"""
import numpy as np
from scipy.optimize import linprog
from path_bounds import prepare,solve


class WindowOracle:
    def __init__(self,p,d,a,mask,eta):
        self.p=np.asarray(p,dtype=np.float64)
        self.mask=np.asarray(mask,dtype=np.float64)
        self.args=prepare(p,d,a)
        self.T=len(p)
        self.wp=np.vstack((np.zeros(self.p.shape[1]),np.cumsum(self.p*self.mask[:,None],axis=0)))
        self.base=solve(*self.args)
        self.floor=self.base[0]/self.T-eta

    def evaluate(self,theta,sign,confidence_floor):
        lam,beta=np.asarray(theta,dtype=float)
        if lam<0 or beta<0:raise ValueError('Negative dual multiplier')
        lp,rp,ld,la,init=self.args
        weighted=sign*self.wp+beta*rp
        out=solve(lp,weighted,ld,la,init,lam=1/self.T,log_weight=lam/self.T)
        chosen=self.p[np.arange(self.T),out[3]]
        confidence=chosen.mean();window=(chosen*self.mask).sum()/self.T
        value=out[0]-lam*self.floor-beta*confidence_floor
        gradient=np.array([out[1]/self.T-self.floor,confidence-confidence_floor])
        intercept=sign*window
        if abs(value-(intercept+np.dot(theta,gradient)))>1e-8:
            raise AssertionError('Support gradient identity failed')
        return float(value),gradient,float(intercept)

    def upper(self,sign,confidence_floor=0.,conditioned=False,max_queries=24):
        upper=float((np.max(sign*self.p,axis=1)*self.mask).sum()/self.T)
        cuts=[];theta=np.array([1.,0.]);best_theta=None;queries=0;last_master=None
        for iteration in range(max_queries):
            value,gradient,intercept=self.evaluate(theta,sign,confidence_floor)
            queries+=1
            if value<upper:upper=value;best_theta=theta.copy()
            cuts.append((gradient,intercept))
            # g.theta+b <= v for each support plane. The master lower-bounds
            # the dual objective over the box, NOT the discrete primal optimum.
            A=np.array([[g[0],g[1],-1.] for g,b in cuts]);b=np.array([-b for g,b in cuts])
            result=linprog([0.,0.,1.],A_ub=A,b_ub=b,
                bounds=[(0.,100.),(0.,100. if conditioned else 0.),(-1.,None)],method='highs')
            if not result.success:break
            last_master=float(result.fun)
            if upper-last_master<=1e-7:break
            proposed=np.maximum(result.x[:2],0.)
            if np.max(np.abs(proposed-theta))<1e-10:break
            theta=proposed
        return dict(upper=upper+1e-9,queries=queries,master_lower_on_box=last_master,
                    best_theta=None if best_theta is None else best_theta.tolist())
