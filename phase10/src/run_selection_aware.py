from pathlib import Path
import hashlib
import json
import time
import numpy as np
from selection_aware_bounds import WindowOracle

ROOT=Path(__file__).resolve().parents[2]
INP=ROOT/'phase10/results/development_inputs_v1'
PREV=ROOT/'phase10/results/functional_bounds_v1'
OUT=ROOT/'phase10/results/selection_aware_windows_v1'
NAMES=['Healthy','Holosystolic','Early-systolic','Mid-systolic']
ETA=.005

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    before=json.loads((PREV/'run.json').read_text())
    for rel,sha in before['sources'].items():assert digest(ROOT/rel)==sha
    for rel,sha in before['output_sha256'].items():assert digest(PREV/rel)==sha
    sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
        ROOT/'phase10/src/selection_aware_bounds.py',ROOT/'phase10/SELECTION_AWARE_PROTOCOL_v1.txt',
        ROOT/'phase10/logs/selection_aware_tests_v1.log',PREV/'run.json']}
    run=dict(status='running',sources=sources,records=24,windows_per_record=4,eta=ETA,
             clinical_labels_read=False,phase_annotations_read=False,final_reserve_arrays_read=0)
    write(OUT/'run.json',run)
    material=json.loads((INP/'materialization.json').read_text())
    old=json.loads((PREV/'topology_bounds.json').read_text())
    old={(r['number'],r['topology']):r for r in old if r['eta']==ETA}
    warm=WindowOracle(np.full((4,2),.4),np.full((2,2),.5),np.eye(2),np.array([1.,0.,0.,0.]),ETA)
    warm.upper(1,.1,conditioned=True)
    rows=[];details=[];started=time.perf_counter()
    for rec in material['records']:
        assert digest(INP/rec['array'])==rec['sha256']
        with np.load(INP/rec['array'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(PREV/f'witnesses_{rec["number"]:02d}.npz',allow_pickle=False) as z:wit={k:z[k] for k in z.files}
        T=len(data['posterior']);edges=np.linspace(0,T,5).astype(int)
        global_bounds={n:old[(rec['number'],n)] for n in NAMES[1:]}
        floors={n:max(global_bounds[k]['lower'] for k in NAMES[1:] if k!=n) for n in NAMES[1:]}
        possible=[n for n in NAMES[1:] if global_bounds[n]['upper']>=floors[n]]
        base_conf={n:float(data[n+'_posterior'][np.arange(T),wit[n+'_base']].astype(np.float64).mean()) for n in NAMES}
        base_winner=max(NAMES[1:],key=lambda n:base_conf[n])
        assert base_winner in possible
        tick_record=time.perf_counter()
        for quarter,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
            mask=np.zeros(T);mask[start:end]=1.
            tick=time.perf_counter();unconditional={};conditional={}
            for name in NAMES:
                p,d,a=(data[name+'_'+s] for s in ['posterior','duration','transition'])
                oracle=WindowOracle(p,d,a,mask,ETA)
                uminus=oracle.upper(-1);uplus=oracle.upper(1)
                unconditional[name]=(-uminus['upper'],uplus['upper'])
                row=dict(number=rec['number'],quarter=quarter,topology=name,
                    start_frame=int(start),end_frame=int(end),unconditional_minus=uminus,unconditional_plus=uplus)
                if name in possible:
                    cminus=oracle.upper(-1,floors[name],conditioned=True)
                    cplus=oracle.upper(1,floors[name],conditioned=True)
                    conditional[name]=(max(-cminus['upper'],-uminus['upper']),min(cplus['upper'],uplus['upper']))
                    row.update(confidence_floor=floors[name],conditioned_minus=cminus,conditioned_plus=cplus)
                details.append(row)
            h=unconditional['Healthy']
            A=(min(unconditional[n][0] for n in NAMES[1:])-h[1],max(unconditional[n][1] for n in NAMES[1:])-h[0])
            B=(min(unconditional[n][0] for n in possible)-h[1],max(unconditional[n][1] for n in possible)-h[0])
            C=(min(conditional[n][0] for n in possible)-h[1],max(conditional[n][1] for n in possible)-h[0])
            baseline=float(((data[base_winner+'_posterior'][np.arange(T),wit[base_winner+'_base']].astype(float)-
                 data['Healthy_posterior'][np.arange(T),wit['Healthy_base']].astype(float))*mask).sum()/T)
            assert A[0]-1e-9<=B[0]<=C[0]+1e-9 and A[1]+1e-9>=B[1]>=C[1]-1e-9
            assert C[0]<=baseline+1e-8 and C[1]>=baseline-1e-8
            rows.append(dict(number=rec['number'],record=rec['record'],quarter=quarter,
                frames=T,start_frame=int(start),end_frame=int(end),possible_winners=possible,
                unconditional_lower=A[0],unconditional_upper=A[1],
                pruned_lower=B[0],pruned_upper=B[1],selection_lower=C[0],selection_upper=C[1],
                zero_tilt_window_contribution=baseline,seconds=time.perf_counter()-tick))
        write(OUT/'window_bounds.json',rows);write(OUT/'oracle_details.json',details)
        print(json.dumps(dict(record=rec['number'],seconds=time.perf_counter()-tick_record)),flush=True)
    reductions=[1-(r['selection_upper']-r['selection_lower'])/(r['pruned_upper']-r['pruned_lower']) for r in rows]
    pruning=[1-(r['pruned_upper']-r['pruned_lower'])/(r['unconditional_upper']-r['unconditional_lower']) for r in rows]
    summary=dict(status='complete',windows=len(rows),records=24,
         median_relative_width_reduction_vs_pruned=float(np.median(reductions)),
         max_relative_width_reduction_vs_pruned=max(reductions),
         windows_more_than_one_percent_tighter=sum(r>.01 for r in reductions),
         median_relative_width_reduction_by_topology_pruning=float(np.median(pruning)),
         median_seconds_per_window=float(np.median([r['seconds'] for r in rows])),
         elapsed_seconds=time.perf_counter()-started,
         interpretation='Adaptive exploratory outer bounds; generic duality; no clinical or novelty claim.')
    write(OUT/'summary.json',summary)
    for rel,sha in sources.items():assert digest(ROOT/rel)==sha
    run.update(status='complete',outputs={p.name:digest(p) for p in OUT.iterdir() if p.name!='run.json'})
    write(OUT/'run.json',run);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
