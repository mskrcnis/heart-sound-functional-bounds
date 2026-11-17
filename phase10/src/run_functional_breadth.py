"""Frozen 48-person, sixteen-condition development extension of functional bounds."""
import time
import pandas as pd
from query_training_common import ROOT,P9,NAMES,Potentials,load,write,digest,verify_hashes,np,Path,hashlib
from path_bounds import FunctionalBounds,solve
from segment_projection_baseline import segment_interval
from cheap_queries_and_winners import interval as fixed_interval
import sys
sys.path.insert(0,str(ROOT/'phase10/tests'))
from test_path_bounds import witness_score

OUT=ROOT/'phase10/results/functional_breadth_v1'
PROTO=ROOT/'phase10/FUNCTIONAL_BREADTH_PROTOCOL_v1.txt'

def classify(lo,hi):return 'positive' if lo>0 else 'nonpositive' if hi<=0 else 'unresolved'
def contrast(bounds):return max(b[0] for b in bounds[1:])-bounds[0][1],max(b[1] for b in bounds[1:])-bounds[0][0]

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);start=time.time()
    first=ROOT/'phase10/results/development_inputs_v1/selection_manifest.json'
    excluded={r['group_id'] for r in load(first)['records']}
    inventory=P9/'results/metadata/record_inventory.csv'
    meta=pd.read_csv(inventory,usecols=['record','patient_id','group_id','site','selected','role'],dtype=str)
    meta=meta[meta.role.eq('pilot_validation') & meta.selected.eq('True') & ~meta.group_id.isin(excluded) & meta.site.isin(['AV','PV','TV','MV'])]
    groups={}
    for r in meta.to_dict('records'):groups.setdefault(r['group_id'],[]).append(r)
    chosen=[]
    for group in sorted(groups,key=lambda g:hashlib.sha256(('phase10-breadth-v1:'+g).encode()).hexdigest())[:48]:
        r=sorted(groups[group],key=lambda r:(['AV','PV','TV','MV'].index(r['site']),r['record']))[0]
        chosen.append(dict(record=r['record'],patient_id=r['patient_id'],group_id=group))
    assert len(chosen)==48 and len({r['group_id'] for r in chosen})==48
    directories={'teacher_float':P9/'results/hsmm_int8_control_v1/cpu_float',
                 'teacher_int8':P9/'results/hsmm_int8_control_v1/cpu_int8'}
    for seed in [20261107,20261108,20261109]:
        for alpha,name in [('0p0','hard'),('1p0','kd')]:
            directories[f'{name}_{seed}']=P9/f'results/physiological_distill_v1/seed_{seed}_alpha_{alpha}/evaluation'
    paths=[PROTO,Path(__file__).resolve(),first,inventory,ROOT/'phase10/src/query_training_common.py',
           ROOT/'phase10/src/path_bounds.py',ROOT/'phase10/src/segment_projection_baseline.py',
           ROOT/'phase10/src/cheap_queries_and_winners.py',ROOT/'phase10/tests/test_path_bounds.py',
           P9/'third_party/parallel-hsmm-murmur/src/segmenter.py',P9/'configs/hsmm_baseline.json']
    selection=[]
    for model,directory in directories.items():
        ledger=directory/'decoder_records.json';paths.append(ledger)
        events={(r['record'],str(r['budget_seconds'])):r for r in load(ledger) if r['role']=='pilot_validation'}
        for budget in [20,'full']:
            for i,r in enumerate(chosen):
                event=events[(r['record'],str(budget))]
                selection.append(dict(r,model=model,budget=budget,number=i,
                    source_directory=str(directory.relative_to(ROOT)),array=event['array'],
                    array_sha256=event['array_file_sha256'],posterior_sha256=event['posterior_raw_sha256']))
    sources={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write(OUT/'selection.json',dict(status='sealed_before_new_array_reads',records=chosen,evaluations=selection,
        sources=sources,distinct_people=48,repeated_conditions=16,selection_used_labels=False,final_reserve_inputs=0))
    run=dict(status='running',sources=sources,selection_sha256=digest(OUT/'selection.json'),eta=.005,
             interpretation='Internal development extension; repeated conditions are not independent people')
    write(OUT/'run.json',run);potentials=Potentials();rows=[];topologies=[];audit_count=0;logerr=conferr=0.
    for index,rec in enumerate(selection):
        source=ROOT/rec['source_directory']/'posteriors'/rec['array']
        assert digest(source)==rec['array_sha256']
        with np.load(source,allow_pickle=False) as z:p=z['posterior'].copy()
        assert hashlib.sha256(p.tobytes()).hexdigest()==rec['posterior_sha256']
        T=len(p);epsilon=.005*T;bounds={};witnesses={};tick=time.time()
        for name,(pm,d,a) in zip(NAMES,potentials(p)):
            model=FunctionalBounds(pm,d,a);base=solve(*model.args);b=model.interval(epsilon,iterations=14)
            slo,shi,edges=segment_interval(*model.args,base[4],model.backward,model.optimum,epsilon)
            flo,fhi=fixed_interval(model.args,base,pm,.005,[1.])
            for side in ['lower','upper']:
                ll,cc=witness_score(pm,d,a,b['segments_'+side]);state=b['witness_'+side]
                actual=float(np.asarray(pm,dtype=np.float64)[np.arange(T),state].mean())
                logerr=max(logerr,abs(ll-b['witness_'+side+'_log_score']))
                conferr=max(conferr,abs(cc-b['feasible_'+side]),abs(actual-cc))
                assert abs(ll-b['witness_'+side+'_log_score'])<1e-8 and abs(actual-cc)<1e-10
                assert ll>=model.optimum-epsilon-1.1e-8 and b['lower']<=cc+1e-10 and b['upper']>=cc-1e-10
                assert slo<=cc+1e-10 and shi>=cc-1e-10 and flo<=cc+1e-10 and fhi>=cc-1e-10
                audit_count+=1
                witnesses[name+'_path_'+side]=b['witness_'+side]
                witnesses[name+'_segments_'+side]=b['segments_'+side]
            clean={k:v for k,v in b.items() if not isinstance(v,np.ndarray)}
            clean.update(topology=name,model=rec['model'],budget=rec['budget'],number=rec['number'],
                record=rec['record'],group_id=rec['group_id'],frames=T,optimum=model.optimum,
                segment_lower=float(slo),segment_upper=float(shi),fixed_lower=flo,fixed_upper=fhi,
                intersect_lower=max(b['lower'],slo-1e-10),intersect_upper=min(b['upper'],shi+1e-10),edges=int(edges))
            bounds[name]=clean;topologies.append(clean)
        result={k:rec[k] for k in ['record','group_id','model','budget','number']};result.update(frames=T,epsilon=epsilon)
        fields={'frame':('projection_lower','projection_upper'),'segment':('segment_lower','segment_upper'),
                'functional':('lower','upper'),'intersect':('intersect_lower','intersect_upper'),
                'fixed':('fixed_lower','fixed_upper'),'feasible':('feasible_lower','feasible_upper')}
        for method,(l,h) in fields.items():
            lo,hi=contrast([(bounds[n][l],bounds[n][h]) for n in NAMES])
            result[method]=dict(lower=lo,upper=hi,sign=classify(lo,hi))
        possible=[];witnessed=[]
        for name in NAMES[1:]:
            others=[n for n in NAMES[1:] if n!=name]
            if bounds[name]['intersect_upper']>=max(bounds[n]['intersect_lower'] for n in others):possible.append(name)
            assignment={n:bounds[n]['feasible_lower'] for n in NAMES[1:]};assignment[name]=bounds[name]['feasible_upper']
            if max(NAMES[1:],key=lambda n:assignment[n])==name:witnessed.append(name)
        assert set(witnessed).issubset(possible)
        result.update(outer_possible_winners=possible,witnessed_winners=witnessed,
            stable_sign_multiple_winners=result['intersect']['sign']!='unresolved' and len(witnessed)>1,
            feasible_both_signs=result['feasible']['lower']<=0<result['feasible']['upper'],seconds=time.time()-tick)
        filename=f'witnesses_{index:04d}.npz';np.savez_compressed(OUT/filename,**witnesses)
        result.update(witnesses=filename,witnesses_sha256=digest(OUT/filename));rows.append(result)
        if (index+1)%48==0:
            write(OUT/'record_bounds.json',rows);write(OUT/'topology_bounds.json',topologies)
            print(f'condition complete: {rec["model"]} {rec["budget"]}; evaluations {index+1}/768',flush=True)
    summaries=[]
    for model in directories:
        for budget in [20,'full']:
            sub=[r for r in rows if r['model']==model and r['budget']==budget];assert len(sub)==48
            widths={method:np.array([r[method]['upper']-r[method]['lower'] for r in sub]) for method in fields if method!='feasible'}
            eligible=widths['segment']>1e-9
            reduction=float(1-np.median(widths['intersect'][eligible]/widths['segment'][eligible]))
            extra=sum(r['segment']['sign']=='unresolved' and r['intersect']['sign']!='unresolved' for r in sub)
            summaries.append(dict(model=model,budget=budget,n=48,
                resolved={k:sum(r[k]['sign']!='unresolved' for r in sub) for k in widths},
                median_width={k:float(np.median(v)) for k,v in widths.items()},
                median_paired_width_reduction_vs_segment=reduction,width_ratio_denominator=int(eligible.sum()),
                additional_resolved_vs_segment=extra,condition_gate_passed=reduction>=.5 and extra>=5,
                witnessed_both_signs=sum(r['feasible_both_signs'] for r in sub),
                witnessed_multiple_winners=sum(len(r['witnessed_winners'])>1 for r in sub),
                stable_sign_multiple_winners=sum(r['stable_sign_multiple_winners'] for r in sub)))
    passed=sum(r['condition_gate_passed'] for r in summaries)
    summary=dict(status='complete',conditions=summaries,conditions_passing=passed,engineering_gate_passed=passed>=12,
        distinct_people=48,record_conditions=len(rows),topology_conditions=len(topologies),witnesses_audited=audit_count,
        max_witness_log_score_error=logerr,max_witness_confidence_error=conferr,missing_or_invalid_inputs=0,
        methodological_novelty_established=False,clinical_labels_used=False,seconds=time.time()-start)
    write(OUT/'summary.json',summary);verify_hashes(sources)
    run.update(status='complete',output_sha256={p.name:digest(p) for p in OUT.iterdir() if p.name!='run.json'})
    write(OUT/'run.json',run);print('BREADTH_COMPLETE '+str(dict(conditions_passing=passed,gate=passed>=12,witnesses=audit_count)))

if __name__=='__main__':main()
