"""Read back every saved breadth witness and independently check report arithmetic."""
from run_functional_breadth import OUT,ROOT,NAMES,Potentials,load,write,digest,verify_hashes,np,Path,solve,witness_score
from path_bounds import prepare

def main():
    out=OUT/'saved_result_audit.json'
    if out.exists():raise FileExistsError(out)
    run=load(OUT/'run.json');assert run['status']=='complete';verify_hashes(run['sources'])
    for rel,sha in run['output_sha256'].items():assert digest(OUT/rel)==sha
    selection=load(OUT/'selection.json');records=load(OUT/'record_bounds.json');top=load(OUT/'topology_bounds.json')
    assert len(records)==len(selection['evaluations'])==768 and len(top)==3072
    assert len({r['group_id'] for r in records})==48
    excluded={r['group_id'] for r in load(ROOT/'phase10/results/development_inputs_v1/selection_manifest.json')['records']}
    assert not excluded & {r['group_id'] for r in records}
    top={(r['model'],str(r['budget']),r['number'],r['topology']):r for r in top}
    assert len(top)==3072
    graph=Potentials();audited=0;maximum_log_error=maximum_confidence_error=0.
    for r,s in zip(records,selection['evaluations']):
        for key in ['record','group_id','model','budget','number']:assert r[key]==s[key]
        pth=ROOT/s['source_directory']/'posteriors'/s['array'];assert digest(pth)==s['array_sha256']
        with np.load(pth,allow_pickle=False) as z:p=z['posterior'].copy()
        assert digest(OUT/r['witnesses'])==r['witnesses_sha256']
        with np.load(OUT/r['witnesses'],allow_pickle=False) as z:witness={k:z[k].copy() for k in z.files}
        bs=[]
        for name,(pm,d,a) in zip(NAMES,graph(p)):
            b=top[(r['model'],str(r['budget']),r['number'],name)];bs.append(b)
            base=solve(*prepare(pm,d,a));assert abs(base[0]-b['optimum'])<1e-10
            for side in ['lower','upper']:
                ll,conf=witness_score(pm,d,a,witness[name+'_segments_'+side])
                actual=np.asarray(pm,dtype=np.float64)[np.arange(len(pm)),witness[name+'_path_'+side]].mean()
                maximum_log_error=max(maximum_log_error,abs(ll-b['witness_'+side+'_log_score']))
                maximum_confidence_error=max(maximum_confidence_error,abs(conf-actual),abs(conf-b['feasible_'+side]))
                assert ll>=base[0]-.005*len(pm)-1.1e-8
                for l,h in [('lower','upper'),('segment_lower','segment_upper'),('fixed_lower','fixed_upper'),('intersect_lower','intersect_upper')]:
                    assert b[l]<=conf+1e-10 and b[h]>=conf-1e-10
                audited+=1
        fields={'frame':('projection_lower','projection_upper'),'segment':('segment_lower','segment_upper'),
                'functional':('lower','upper'),'intersect':('intersect_lower','intersect_upper'),
                'fixed':('fixed_lower','fixed_upper'),'feasible':('feasible_lower','feasible_upper')}
        for method,(l,h) in fields.items():
            lo=max(b[l] for b in bs[1:])-bs[0][h];hi=max(b[h] for b in bs[1:])-bs[0][l]
            sign='positive' if lo>0 else 'nonpositive' if hi<=0 else 'unresolved'
            assert abs(lo-r[method]['lower'])<1e-12 and abs(hi-r[method]['upper'])<1e-12 and sign==r[method]['sign']
        witnessed=[];possible=[]
        for j in range(1,4):
            values=[b['feasible_lower'] for b in bs[1:]];values[j-1]=bs[j]['feasible_upper']
            if int(np.argmax(values))==j-1:witnessed.append(NAMES[j])
            if bs[j]['intersect_upper']>=max(bs[k]['intersect_lower'] for k in range(1,4) if k!=j):possible.append(NAMES[j])
        assert witnessed==r['witnessed_winners'] and possible==r['outer_possible_winners']
        assert r['stable_sign_multiple_winners']==(r['intersect']['sign']!='unresolved' and len(witnessed)>1)
    assert audited==6144 and maximum_log_error<1e-8 and maximum_confidence_error<1e-10
    summary=load(OUT/'summary.json');passed=0
    for condition in summary['conditions']:
        sub=[r for r in records if r['model']==condition['model'] and r['budget']==condition['budget']]
        assert len(sub)==48
        pairs=[(r['intersect']['upper']-r['intersect']['lower'])/(r['segment']['upper']-r['segment']['lower']) for r in sub if r['segment']['upper']-r['segment']['lower']>1e-9]
        gain=1-float(np.median(pairs));extra=sum(r['segment']['sign']=='unresolved' and r['intersect']['sign']!='unresolved' for r in sub)
        assert abs(gain-condition['median_paired_width_reduction_vs_segment'])<1e-12
        assert extra==condition['additional_resolved_vs_segment']
        for key,value in condition['resolved'].items():assert value==sum(r[key]['sign']!='unresolved' for r in sub)
        passed+=gain>=.5 and extra>=5
    assert passed==summary['conditions_passing'] and (passed>=12)==summary['engineering_gate_passed']
    write(out,dict(status='passed',record_conditions=768,distinct_people=48,topology_conditions=3072,
        witnesses_rescored=audited,max_log_score_error=maximum_log_error,max_confidence_error=maximum_confidence_error,
        all_source_and_output_hashes_passed=True,conditions_passing=int(passed),engineering_gate_passed=passed>=12,
        sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),OUT/'run.json',OUT/'summary.json']},
        scope='All saved witnesses, base optimum values, interval arithmetic and winner summaries; no clinical validation'))
    print('BREADTH_AUDIT_PASSED '+str(dict(witnesses=audited,conditions=int(passed))))

if __name__=='__main__':main()
