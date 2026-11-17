from pathlib import Path
import hashlib
import json
import time
import numpy as np
from path_bounds import prepare,solve,LOG_TOL,CONF_TOL

ROOT=Path(__file__).resolve().parents[2]
INP=ROOT/'phase10/results/development_inputs_v1'
PREV=ROOT/'phase10/results/functional_bounds_v1'
OUT=ROOT/'phase10/results/cheap_queries_and_winners_v1'
NAMES=['Healthy','Holosystolic','Early-systolic','Mid-systolic']
ETAS=[.001,.005,.01,.05]
GRIDS={'one':[1.],'three':[.3,1.,3.],'six':[.1,.3,1.,3.,10.,30.]}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def classify(lo,hi):return 'positive' if lo>0 else 'nonpositive' if hi<=0 else 'unresolved'

def interval(args,base,p,eta,lambdas):
    p=np.asarray(p,dtype=np.float64)
    T=len(p);lo=float(p.min(axis=1).mean());hi=float(p.max(axis=1).mean())
    for magnitude in lambdas:
        for sign in [-1.,1.]:
            lam=magnitude*sign;out=solve(*args,lam=lam)
            v=(out[0]-base[0]+eta*T+LOG_TOL)/(lam*T)
            if sign>0:hi=min(hi,v)
            else:lo=max(lo,v)
    return lo-CONF_TOL,hi+CONF_TOL

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    previous=json.loads((PREV/'run.json').read_text())
    for rel,sha in previous['sources'].items():assert digest(ROOT/rel)==sha
    for rel,sha in previous['output_sha256'].items():assert digest(PREV/rel)==sha
    material=json.loads((INP/'materialization.json').read_text())
    top=json.loads((PREV/'topology_bounds.json').read_text())
    top={(r['number'],r['topology'],r['eta']):r for r in top}
    old=json.loads((PREV/'record_bounds.json').read_text());old={(r['number'],r['eta']):r for r in old}
    rows=[];winners=[]
    solve(*prepare(np.full((4,2),.4),np.full((2,2),.5),np.eye(2)))
    for rec in material['records']:
        with np.load(INP/rec['array'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        for eta in ETAS:
            marg={n:top[(rec['number'],n,eta)] for n in NAMES[1:]}
            possible=[];witnessed=[]
            for name in NAMES[1:]:
                others=[n for n in NAMES[1:] if n!=name]
                if marg[name]['upper']>=max(marg[n]['lower'] for n in others):possible.append(name)
                assignment={n:marg[n]['feasible_lower'] for n in NAMES[1:]}
                assignment[name]=marg[name]['feasible_upper']
                if max(NAMES[1:],key=lambda n:assignment[n])==name:witnessed.append(name)
            assert set(witnessed).issubset(possible)
            winners.append(dict(number=rec['number'],eta=eta,outer_possible=possible,
                 witnessed_possible=witnessed,site_score_sign=old[(rec['number'],eta)]['sign']))
            for key,grid in GRIDS.items():
                per=[];begin=time.perf_counter()
                for name in NAMES:
                    p=data[name+'_posterior'];args=prepare(p,data[name+'_duration'],data[name+'_transition'])
                    base=solve(*args)
                    lo,hi=interval(args,base,p,eta,grid)
                    known=top[(rec['number'],name,eta)]
                    assert lo<=known['feasible_lower']+1e-9 and hi>=known['feasible_upper']-1e-9
                    per.append((lo,hi))
                elapsed=time.perf_counter()-begin
                lo=max(x[0] for x in per[1:])-per[0][1]
                hi=max(x[1] for x in per[1:])-per[0][0]
                rows.append(dict(number=rec['number'],eta=eta,query_set=key,dp_calls=4*(1+2*len(grid)),
                  lower=lo,upper=hi,sign=classify(lo,hi),seconds=elapsed,
                  full_outer_width=old[(rec['number'],eta)]['upper']-old[(rec['number'],eta)]['lower']))
    summary=[]
    for eta in ETAS:
        for key in GRIDS:
            sub=[r for r in rows if r['eta']==eta and r['query_set']==key]
            summary.append(dict(eta=eta,query_set=key,dp_calls=sub[0]['dp_calls'],
                sign_certified=sum(r['sign']!='unresolved' for r in sub),
                median_width=float(np.median([r['upper']-r['lower'] for r in sub])),
                median_width_ratio_vs_full=float(np.median([(r['upper']-r['lower'])/r['full_outer_width'] for r in sub])),
                median_seconds=float(np.median([r['seconds'] for r in sub]))))
    winner_summary=[]
    for eta in ETAS:
        sub=[r for r in winners if r['eta']==eta]
        winner_summary.append(dict(eta=eta,
             outer_single_winner=sum(len(r['outer_possible'])==1 for r in sub),
             witnessed_multiple_winners=sum(len(r['witnessed_possible'])>1 for r in sub),
             stable_score_but_witnessed_multiple_winners=sum(r['site_score_sign']!='unresolved' and len(r['witnessed_possible'])>1 for r in sub)))
    write(OUT/'record_query_results.json',rows);write(OUT/'record_topology_winners.json',winners)
    result=dict(status='complete',query_results=summary,winner_results=winner_summary,
        clinical_labels_read=False,final_reserve_arrays_read=0,
        timing_scope='One record, four topologies, one eta; potential preparation + base and fixed dual solves. JIT excluded. No neural/audio frontend. Single observational pass.',
        sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),ROOT/'phase10/FOLLOWUP_v1.txt',PREV/'run.json']})
    write(OUT/'summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
