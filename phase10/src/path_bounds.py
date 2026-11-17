# Public release adaptation notice, 2026-11-17: license notice added;
# numerical implementation unchanged. Derived boundary semantics: University
# of Cambridge, 2022. GAP Available Source License v1.0; see LICENSE.
"""Exploratory downstream-functional bounds on the legacy HSMM boundary DAG.

Boundary semantics adapted from Cambridge's 2022 parallel-HSMM implementation
(GAP Available Source License v1.0); see ../../phase9/HSMM_LICENSE.md.
This independent float64 implementation is NOT the frozen float32 decoder.
Weak-duality and max-marginal operations are established algorithms.
"""
from __future__ import annotations

import numpy as np
from numba import njit

LOG_TOL = 1e-8
CONF_TOL = 1e-10
TINY = float(np.finfo(np.float32).tiny)


def prepare(posterior, duration, transition):
    p = np.asarray(posterior, dtype=np.float64)
    d = np.asarray(duration, dtype=np.float64)
    a = np.asarray(transition, dtype=np.float64)
    if p.ndim != 2 or not len(p) or d.ndim != 2 or a.shape != (p.shape[1],)*2:
        raise ValueError('Invalid dimensions')
    if d.shape[1] != p.shape[1] or not len(d):
        raise ValueError('Invalid durations')
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (p,d,a)):
        raise ValueError('Nonfinite or negative potentials')
    if (p[0] <= 0).any():
        raise ValueError('Prototype requires positive initialization potentials')
    lp = np.vstack((np.zeros(p.shape[1]), np.cumsum(np.log(p+TINY),axis=0)))
    rp = np.vstack((np.zeros(p.shape[1]), np.cumsum(p,axis=0)))
    return (lp, rp, np.log(d+TINY), np.log(a+TINY), np.log(p[0]))


@njit(cache=True)
def solve(lp, rp, ld, la, init, lam=0., log_weight=1.):
    """Maximize log_weight*L+lam*R; retain the actual L,R and path."""
    T, N = lp.shape[0]-1, lp.shape[1]
    D = ld.shape[0]
    F = np.full((T+D,N), -np.inf)
    best_in = np.full((T+D,N), -np.inf)
    prev_in = np.zeros((T+D,N),np.int64)
    prev_t = np.zeros((T+D,N),np.int64)
    prev_s = np.zeros((T+D,N),np.int64)
    prev_d = np.zeros((T+D,N),np.int64)
    F[0] = log_weight*init
    for t in range(T+D):
        if t > 0:
            end = min(t,T)
            for s in range(N):
                for d in range(1,D+1):
                    u = max(0,min(t-d,T-1))
                    edge = log_weight*(lp[end,s]-lp[u,s]+ld[d-1,s])
                    edge += lam*(rp[end,s]-rp[u,s])
                    val = best_in[u,s]+edge
                    if val > F[t,s]:
                        F[t,s] = val
                        prev_t[t,s],prev_s[t,s],prev_d[t,s]=u,prev_in[u,s],d
        for s in range(N):
            for i in range(N):
                val = F[t,i]+log_weight*la[i,s]
                if val > best_in[t,s]:
                    best_in[t,s],prev_in[t,s]=val,i
    val,bt,bs = -np.inf,0,0
    for t in range(T,T+D):
        for s in range(N):
            if F[t,s] > val:
                val,bt,bs = F[t,s],t,s
    states = np.full(T,-1,np.int64)
    segments = np.full((T+D,5),-1,np.int64)
    segment_count=0
    ll,rr=0.,0.
    t,s=bt,bs
    while t > 0:
        u,i,d=prev_t[t,s],prev_s[t,s],prev_d[t,s]
        end=min(t,T)
        segments[segment_count]=np.array([u,t,i,s,d])
        segment_count+=1
        states[u:end]=s
        ll+=lp[end,s]-lp[u,s]+ld[d-1,s]+la[i,s]
        rr+=rp[end,s]-rp[u,s]
        t,s=u,i
    ll+=init[s]
    return val,ll,rr,states,F,best_in,segments[:segment_count].copy()


@njit(cache=True)
def projections(lp,rp,ld,la,init,F,best_in):
    """Exact frame-state max-marginals on the specified boundary DAG."""
    T,N=lp.shape[0]-1,lp.shape[1]
    D=ld.shape[0]
    B=np.full((T+D,N),-np.inf)
    B[T:,:]=0.
    # Each edge has u<t, so reverse endpoint order is topological.
    for t in range(T+D-1,0,-1):
        end=min(t,T)
        for s in range(N):
            for d in range(1,D+1):
                u=max(0,min(t-d,T-1))
                value=lp[end,s]-lp[u,s]+ld[d-1,s]+B[t,s]
                for i in range(N):
                    B[u,i]=max(B[u,i],la[i,s]+value)
    size=1
    while size<T: size*=2
    tree=np.full((N,2*size),-np.inf)
    for t in range(1,T+D):
        end=min(t,T)
        for s in range(N):
            for d in range(1,D+1):
                u=max(0,min(t-d,T-1))
                val=best_in[u,s]+lp[end,s]-lp[u,s]+ld[d-1,s]+B[t,s]
                # Range maximum tags. No enumeration of individual frames.
                left,right=u+size,end+size
                while left<right:
                    if left%2:
                        tree[s,left]=max(tree[s,left],val);left+=1
                    if right%2:
                        right-=1;tree[s,right]=max(tree[s,right],val)
                    left//=2;right//=2
    for s in range(N):
        for j in range(1,size):
            tree[s,2*j]=max(tree[s,2*j],tree[s,j])
            tree[s,2*j+1]=max(tree[s,2*j+1],tree[s,j])
    return tree[:,size:size+T].T.copy(),B


class FunctionalBounds:
    def __init__(self, posterior, duration, transition):
        self.p=np.asarray(posterior,dtype=np.float64)
        self.args=prepare(posterior,duration,transition)
        self.T=len(self.p)
        base=solve(*self.args)
        self.optimum=base[0]
        self.base_path=base[3]
        self.base_confidence=base[2]/self.T
        self.max_marginal,self.backward=projections(*self.args,base[4],base[5])
        self.cache={0.:(*base[:4],base[6])}

    def support(self, lam):
        key=float(lam)
        if key not in self.cache:
            out=solve(*self.args,lam=key)
            self.cache[key]=(*out[:4],out[6])
        return self.cache[key]

    def interval(self, epsilon, iterations=14):
        if epsilon<0: raise ValueError('Negative tolerance')
        allowed=self.max_marginal>=self.optimum-epsilon-LOG_TOL
        if not allowed.any(axis=1).all(): raise AssertionError('Empty projection')
        proj_lo=float(np.where(allowed,self.p,np.inf).min(axis=1).mean())
        proj_hi=float(np.where(allowed,self.p,-np.inf).max(axis=1).mean())
        low,high=proj_lo,proj_hi
        # Signed support search: find feasible and infeasible path-score losses.
        for sign in (-1.,1.):
            left,right=0.,1.
            while right<1048576:
                q=self.support(sign*right)
                if self.optimum-q[1] > epsilon+LOG_TOL: break
                left,right=right,right*2
            for _ in range(iterations):
                mid=(left+right)/2
                q=self.support(sign*mid)
                if self.optimum-q[1] <= epsilon+LOG_TOL: left=mid
                else: right=mid
        inner_lo=inner_hi=self.base_confidence
        witness_lo=witness_hi=self.base_path
        segments_lo=segments_hi=self.cache[0.][4]
        ll_lo=ll_hi=self.cache[0.][1]
        for lam,q in self.cache.items():
            _,ll,rr,path,segments=q
            if self.optimum-ll<=epsilon+LOG_TOL:
                if rr/self.T<inner_lo:
                    inner_lo,witness_lo,segments_lo,ll_lo=rr/self.T,path,segments,ll
                if rr/self.T>inner_hi:
                    inner_hi,witness_hi,segments_hi,ll_hi=rr/self.T,path,segments,ll
            if lam:
                # Guard is scaled with lambda; no outward rounding claim.
                val=(q[0]-self.optimum+epsilon+LOG_TOL)/(lam*self.T)
                if lam>0:high=min(high,val)
                else:low=max(low,val)
        low-=CONF_TOL;high+=CONF_TOL
        if low>inner_lo+1e-9 or high<inner_hi-1e-9:
            raise AssertionError('Outer bound excludes a feasible witness')
        return dict(epsilon=epsilon,lower=low,upper=high,
                    projection_lower=proj_lo,projection_upper=proj_hi,
                    feasible_lower=inner_lo,feasible_upper=inner_hi,
                    lower_gap=inner_lo-low,upper_gap=high-inner_hi,
                    ambiguous_frame_fraction=float((allowed.sum(axis=1)>1).mean()),
                    solver_calls=len(self.cache),witness_lower=witness_lo.copy(),
                    witness_upper=witness_hi.copy(),segments_lower=segments_lo.copy(),
                    segments_upper=segments_hi.copy(),witness_lower_log_score=ll_lo,
                    witness_upper_log_score=ll_hi)
