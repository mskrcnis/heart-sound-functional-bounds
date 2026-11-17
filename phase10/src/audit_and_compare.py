"""Rescore all saved witnesses and add a stronger post-initial-result comparator.

All v1 results are preserved. This follow-up is explicitly adaptive method
development, using the same 24 label-free records and all declared tolerances.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from path_bounds import FunctionalBounds,solve
from segment_projection_baseline import segment_interval

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'phase10/tests'))
from test_path_bounds import witness_score,enumerate_paths
INP=ROOT/'phase10/results/development_inputs_v1'
PREV=ROOT/'phase10/results/functional_bounds_v1'
OUT=ROOT/'phase10/results/strong_baseline_audit_v1'
NAMES=['Healthy','Holosystolic','Early-systolic','Mid-systolic']
ETAS=[0.,.001,.005,.01,.05]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sign(lo,hi):return 'positive' if lo>0 else 'nonpositive' if hi<=0 else 'unresolved'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    previous=json.loads((PREV/'run.json').read_text())
    for rel,sha in previous['sources'].items():
        assert digest(ROOT/rel)==sha,rel
    for rel,sha in previous['output_sha256'].items():
        assert digest(PREV/rel)==sha,rel
    sourcepaths=[Path(__file__).resolve(),ROOT/'phase10/src/segment_projection_baseline.py',
                 ROOT/'phase10/tests/test_path_bounds.py',PREV/'run.json']
    sources={str(p.relative_to(ROOT)):digest(p) for p in sourcepaths}
    # Exhaustive containment check of the new comparator on 30 tiny graphs.
    rng=np.random.default_rng(20261109)
    for case in range(30):
        T=1+case%4;D=1+case%3
        p=rng.dirichlet([1,1,1],T)[:,:2]
        d=rng.dirichlet(np.ones(D),2).T;a=rng.dirichlet([1,1],2)
        model=FunctionalBounds(p,d,a);base=solve(*model.args)
        paths=enumerate_paths(p,d,a)
        for eps in [0.,.1,1.,10.]:
            lo,hi,_=segment_interval(*model.args,base[4],model.backward,model.optimum,eps)
            c=[r/T for l,r,_ in paths if l>=model.optimum-eps-1e-10]
            assert lo<=min(c)+1e-9 and hi>=max(c)-1e-9
    material=json.loads((INP/'materialization.json').read_text())
    oldrows=json.loads((PREV/'record_bounds.json').read_text())
    top=json.loads((PREV/'topology_bounds.json').read_text())
    top={(r['number'],r['topology'],r['eta']):r for r in top}
    rows=[];audited=0;max_logerr=max_conferr=0.;start=time.perf_counter()
    for rec in material['records']:
        assert digest(INP/rec['array'])==rec['sha256']
        with np.load(INP/rec['array'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(PREV/f'witnesses_{rec["number"]:02d}.npz',allow_pickle=False) as z:wit={k:z[k] for k in z.files}
        bounds={eta:{} for eta in ETAS};T=len(data['posterior'])
        for name in NAMES:
            p,d,a=(data[name+'_'+s] for s in ['posterior','duration','transition'])
            model=FunctionalBounds(p,d,a);base=solve(*model.args)
            for eta in ETAS:
                old=top[(rec['number'],name,eta)]
                for side in ['lower','upper']:
                    segments=wit[name+'_'+str(eta)+'_segments_'+side]
                    ll,cc=witness_score(p,d,a,segments)
                    state=wit[name+'_'+str(eta)+'_witness_'+side]
                    score=float(p[np.arange(T),state].astype(np.float64).mean())
                    max_logerr=max(max_logerr,abs(ll-old['witness_'+side+'_log_score']))
                    max_conferr=max(max_conferr,abs(cc-old['feasible_'+side]),abs(cc-score))
                    assert abs(ll-old['witness_'+side+'_log_score'])<1e-8
                    assert abs(cc-old['feasible_'+side])<1e-10 and abs(cc-score)<1e-10
                    assert ll>=model.optimum-eta*T-1.1e-8
                    assert old['lower']<=cc+1e-10 and old['upper']>=cc-1e-10
                    audited+=1
                tick=time.perf_counter()
                lo,hi,edges=segment_interval(*model.args,base[4],model.backward,model.optimum,eta*T)
                elapsed=time.perf_counter()-tick
                assert lo<=old['feasible_lower']+1e-10 and hi>=old['feasible_upper']-1e-10
                bounds[eta][name]=dict(lower=lo,upper=hi,seconds=elapsed,edges=edges)
        for eta in ETAS:
            b=bounds[eta];h=b['Healthy'];m=[b[n] for n in NAMES[1:]]
            lo=max(x['lower'] for x in m)-h['upper'];hi=max(x['upper'] for x in m)-h['lower']
            old=next(r for r in oldrows if r['number']==rec['number'] and r['eta']==eta)
            final_lo=max(old['lower'],lo-2e-10);final_hi=min(old['upper'],hi+2e-10)
            assert final_lo<=old['feasible_lower']+1e-9 and final_hi>=old['feasible_upper']-1e-9
            rows.append(dict(number=rec['number'],record=rec['record'],eta=eta,
              segment_lower=lo,segment_upper=hi,segment_sign=sign(lo,hi),
              dual_lower=old['lower'],dual_upper=old['upper'],
              intersected_lower=final_lo,intersected_upper=final_hi,intersected_sign=sign(final_lo,final_hi),
              feasible_lower=old['feasible_lower'],feasible_upper=old['feasible_upper'],
              seconds_segment_dps=sum(x['seconds'] for x in b.values())))
    summary=[]
    for eta in ETAS:
        sub=[r for r in rows if r['eta']==eta]
        ratios=[(r['intersected_upper']-r['intersected_lower'])/(r['segment_upper']-r['segment_lower'])
                for r in sub if r['segment_upper']-r['segment_lower']>1e-9]
        summary.append(dict(eta=eta,segment_certified=sum(r['segment_sign']!='unresolved' for r in sub),
            dual_and_segment_certified=sum(r['intersected_sign']!='unresolved' for r in sub),
            additional_vs_segment=sum(r['segment_sign']=='unresolved' and r['intersected_sign']!='unresolved' for r in sub),
            median_width_reduction_vs_segment=None if not ratios else float(1-np.median(ratios)),
            median_segment_width=float(np.median([r['segment_upper']-r['segment_lower'] for r in sub])),
            median_dual_segment_width=float(np.median([r['intersected_upper']-r['intersected_lower'] for r in sub])),
            median_segment_seconds=float(np.median([r['seconds_segment_dps'] for r in sub]))))
    audit=dict(status='passed',witnesses_rescored=audited,max_log_score_error=max_logerr,
      max_confidence_error=max_conferr,synthetic_instances=30,synthetic_tolerances=4,
      source_hashes_passed=True,comparison_is_post_initial_results=True,results=summary,
      elapsed_seconds=time.perf_counter()-start,sources=sources)
    write(OUT/'record_comparison.json',rows);write(OUT/'audit.json',audit)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
