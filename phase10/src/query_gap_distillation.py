# Public release adaptation notice, 2026-11-17: license notice added;
# numerical implementation unchanged. Derived boundary semantics: University
# of Cambridge, 2022. GAP Available Source License v1.0; see LICENSE.
"""Differentiable fixed-graph support-gap target; no fitting is performed.

Uses the independently checked phase10 legacy-boundary DAG (GAP ASL v1.0;
see ../../phase9/HSMM_LICENSE.md). The max-envelope differentiation rule is
standard. No novelty or unseen-input certificate follows from this module.
"""
import numpy as np
import torch
from path_bounds import prepare,solve,TINY


def path_value(p,d,a,path,segments,reward_weights):
    """Reconstruct on torch tensors; optimized path indices are held fixed."""
    T=len(p);idx=torch.as_tensor(path,dtype=torch.long,device=p.device)
    observed=p[torch.arange(T,device=p.device),idx]
    initial=int(segments[-1,2])
    log_score=torch.log(p[0,initial])+torch.log(observed+TINY).sum()
    edges=torch.as_tensor(np.asarray(segments),dtype=torch.long,device=p.device)
    log_score=log_score+torch.log(d[edges[:,4]-1,edges[:,3]]+TINY).sum()
    log_score=log_score+torch.log(a[edges[:,2],edges[:,3]]+TINY).sum()
    reward=(observed*reward_weights).sum()
    return log_score,reward


def query_gaps(posterior,duration,transition,lambdas=(-1.,1.),reward_weights=None):
    """Return differentiable G(lambda) for a fixed topology and potentials.

DP runs on detached CPU float64 values; selected path scores/rewards are
reconstructed on the original device so gradients reach posterior emissions.
Duration/transition gradients are intentionally unsupported in this prototype.
"""
    if not lambdas or any(lam==0 or not np.isfinite(lam) for lam in lambdas):
        raise ValueError('Queries must be finite and nonzero')
    p=posterior.to(dtype=torch.float64)
    d=torch.as_tensor(duration,dtype=torch.float64,device=p.device).detach()
    a=torch.as_tensor(transition,dtype=torch.float64,device=p.device).detach()
    w=torch.ones(len(p),dtype=torch.float64,device=p.device) if reward_weights is None else torch.as_tensor(reward_weights,dtype=torch.float64,device=p.device)
    if w.shape!=(len(p),) or not torch.isfinite(w).all():raise ValueError('Invalid reward weights')
    if w.requires_grad:raise ValueError('Reward weights must be fixed')
    array=p.detach().cpu().numpy();dn=d.cpu().numpy();an=a.cpu().numpy()
    lp,rp,ld,la,init=prepare(array,dn,an)
    weighted_rp=np.vstack((np.zeros(array.shape[1]),np.cumsum(array*w.cpu().numpy()[:,None],axis=0)))
    base=solve(lp,weighted_rp,ld,la,init)
    l0,_=path_value(p,d,a,base[3],base[6],w)
    values=[]
    for lam in lambdas:
        candidate=solve(lp,weighted_rp,ld,la,init,lam=float(lam))
        ll,rr=path_value(p,d,a,candidate[3],candidate[6],w)
        values.append((ll+float(lam)*rr-l0)/(float(lam)*len(p)))
    return torch.stack(values)


def envelopes(gaps,lambdas,eta):
    """Finite-query bounds only; no model-specific clipping or projections."""
    if eta<0:raise ValueError('Negative tolerance')
    positive=[i for i,x in enumerate(lambdas) if x>0]
    negative=[i for i,x in enumerate(lambdas) if x<0]
    if not positive or not negative:raise ValueError('Both query signs required')
    shifted=gaps+gaps.new_tensor([eta/float(x) for x in lambdas])
    return shifted[negative].max(),shifted[positive].min()
