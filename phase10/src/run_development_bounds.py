"""Fixed phase10 label-free diagnostic; run with conda pytorch_env Python."""
from pathlib import Path
import hashlib
import json
import time
import platform
import numpy as np
import numba
from path_bounds import FunctionalBounds

ROOT=Path(__file__).resolve().parents[2]
INP=ROOT/'phase10/results/development_inputs_v1'
OUT=ROOT/'phase10/results/functional_bounds_v1'
NAMES=['Healthy','Holosystolic','Early-systolic','Mid-systolic']
ETAS=[0.,.001,.005,.01,.05]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sign(lower,upper):return 'positive' if lower>0 else 'nonpositive' if upper<=0 else 'unresolved'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    material=json.loads((INP/'materialization.json').read_text())
    if digest(INP/'selection_manifest.json')!=material['selection_sha256']:raise AssertionError('Selection changed')
    sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
              ROOT/'phase10/src/path_bounds.py',ROOT/'phase10/PROTOCOL_v1.txt',
              INP/'materialization.json',INP/'selection_manifest.json']}
    run=dict(status='running',sources=sources,python=platform.python_version(),
             numpy=np.__version__,numba=numba.__version__,etas=ETAS,records=24,
             claims='Exploratory fixed-posterior computational bounds; no outcome assessment')
    write(OUT/'run.json',run)
    # Warm JIT before diagnostic wall times.
    warm=FunctionalBounds(np.full((4,2),.4),np.ones((2,2))*.5,np.eye(2))
    warm.interval(.1)
    rows=[];topologies=[];start=time.perf_counter()
    for row in material['records']:
        path=INP/row['array']
        if digest(path)!=row['sha256']:raise AssertionError('Materialized input changed')
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        T=len(arrays['posterior']);by_eta={e:{} for e in ETAS};bases={};witnesses={}
        record_start=time.perf_counter()
        for name in NAMES:
            tick=time.perf_counter()
            model=FunctionalBounds(arrays[name+'_posterior'],arrays[name+'_duration'],arrays[name+'_transition'])
            construction=time.perf_counter()-tick
            bases[name]=dict(confidence=model.base_confidence,log_score=model.optimum)
            witnesses[name+'_base']=model.base_path
            for eta in ETAS:
                tick=time.perf_counter();result=model.interval(eta*T)
                elapsed=time.perf_counter()-tick
                for field in ['witness_lower','witness_upper','segments_lower','segments_upper']:
                    witnesses[name+'_'+str(eta)+'_'+field]=result.pop(field)
                result.update(number=row['number'],record=row['record'],group_id=row['group_id'],
                              topology=name,eta=eta,frames=T,seconds=elapsed,construction_seconds=construction)
                by_eta[eta][name]=result;topologies.append(result)
        best=max(NAMES[1:],key=lambda n:bases[n]['confidence'])
        score=bases[best]['confidence']-bases['Healthy']['confidence']
        old=row['legacy'];legacy_best=old['murmur_model']
        comparison=dict(score_difference=score-old['score'],topology_match=best==legacy_best,
                        healthy_changed_frames=int((witnesses['Healthy_base']!=arrays['legacy_healthy']).sum()),
                        legacy_murmur_topology_changed_frames=int((witnesses[legacy_best+'_base']!=arrays['legacy_murmur']).sum()))
        for eta in ETAS:
            b=by_eta[eta];h=b['Healthy'];m=[b[n] for n in NAMES[1:]]
            lo=max(v['lower'] for v in m)-h['upper']
            hi=max(v['upper'] for v in m)-h['lower']
            plo=max(v['projection_lower'] for v in m)-h['projection_upper']
            phi=max(v['projection_upper'] for v in m)-h['projection_lower']
            ilo=max(v['feasible_lower'] for v in m)-h['feasible_upper']
            ihi=max(v['feasible_upper'] for v in m)-h['feasible_lower']
            result=dict(number=row['number'],record=row['record'],group_id=row['group_id'],eta=eta,
                        frames=T,base_score=score,base_topology=best,lower=lo,upper=hi,
                        projection_lower=plo,projection_upper=phi,feasible_lower=ilo,feasible_upper=ihi,
                        sign=sign(lo,hi),projection_sign=sign(plo,phi),
                        feasible_both_signs=ilo<=0<ihi,comparison=comparison,
                        seconds_per_record_all_topologies_all_tolerances=time.perf_counter()-record_start)
            rows.append(result)
        np.savez_compressed(OUT/f'witnesses_{row["number"]:02d}.npz',**witnesses)
        write(OUT/'record_bounds.json',rows);write(OUT/'topology_bounds.json',topologies)
        print(json.dumps(dict(record=row['number'],seconds=round(time.perf_counter()-record_start,3),
                             score=score,legacy=comparison)),flush=True)
    for rel,sha in sources.items():
        if digest(ROOT/rel)!=sha:raise AssertionError('Source changed during run')
    summaries=[]
    for eta in ETAS:
        part=[r for r in rows if r['eta']==eta]
        ratio=[(r['upper']-r['lower'])/(r['projection_upper']-r['projection_lower'])
               for r in part if r['projection_upper']-r['projection_lower']>1e-9]
        summaries.append(dict(eta=eta,records=len(part),certified=sum(r['sign']!='unresolved' for r in part),
            projected_certified=sum(r['projection_sign']!='unresolved' for r in part),
            additional_certified=sum(r['sign']!='unresolved' and r['projection_sign']=='unresolved' for r in part),
            feasible_both_signs=sum(r['feasible_both_signs'] for r in part),
            median_width_reduction=None if not ratio else float(1-np.median(ratio)),
            median_outer_width=float(np.median([r['upper']-r['lower'] for r in part])),
            median_projection_width=float(np.median([r['projection_upper']-r['projection_lower'] for r in part]))))
    summary=dict(results=summaries,elapsed_seconds=time.perf_counter()-start,
                 zero_tilt_legacy_max_abs_score_difference=max(abs(r['comparison']['score_difference']) for r in rows),
                 zero_tilt_legacy_records_with_any_path_mismatch=sum(
                     r['comparison']['healthy_changed_frames']>0 or r['comparison']['legacy_murmur_topology_changed_frames']>0
                     or not r['comparison']['topology_match'] for r in rows if r['eta']==0))
    mid=next(r for r in summaries if r['eta']==.005)
    summary['engineering_gate_passed']=(mid['median_width_reduction']>=.25 and
        any(r['additional_certified']>=2 or r['feasible_both_signs']>=2 for r in summaries if r['eta']>0))
    summary['methodological_novelty_established']=False
    write(OUT/'summary.json',summary)
    run.update(status='complete',elapsed_seconds=summary['elapsed_seconds'],
               output_sha256={p.name:digest(p) for p in OUT.iterdir() if p.name!='run.json'})
    write(OUT/'run.json',run);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
