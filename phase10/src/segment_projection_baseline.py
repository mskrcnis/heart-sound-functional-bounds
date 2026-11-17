# Public release adaptation notice, 2026-11-17: license notice added;
# numerical implementation unchanged. Derived boundary semantics: University
# of Cambridge, 2022. GAP Available Source License v1.0; see LICENSE.
"""Stronger established comparator: optimize confidence on a projected edge DAG.

Keep each segment/transition edge occurring in some likelihood-near-optimal
complete path, then optimize confidence over combinations of these edges.
This remains an outer relaxation: their combined deficit may exceed epsilon.
Legacy graph adaptation: GAP ASL v1.0, see ../../phase9/HSMM_LICENSE.md.
"""
import numpy as np
from numba import njit
from path_bounds import LOG_TOL


@njit(cache=True)
def segment_interval(lp,rp,ld,la,init,F,B,optimum,epsilon):
    T,N=lp.shape[0]-1,lp.shape[1];D=ld.shape[0]
    low=np.full((T+D,N),np.inf);high=np.full((T+D,N),-np.inf)
    low[0]=0.;high[0]=0.
    edge_count=0
    for t in range(1,T+D):
        end=min(t,T)
        for s in range(N):
            for d in range(1,D+1):
                u=max(0,min(t-d,T-1))
                seg=lp[end,s]-lp[u,s]+ld[d-1,s]
                reward=rp[end,s]-rp[u,s]
                for i in range(N):
                    if F[u,i]+la[i,s]+seg+B[t,s]>=optimum-epsilon-LOG_TOL:
                        edge_count+=1
                        low[t,s]=min(low[t,s],low[u,i]+reward)
                        high[t,s]=max(high[t,s],high[u,i]+reward)
    return np.min(low[T:])/T,np.max(high[T:])/T,edge_count
