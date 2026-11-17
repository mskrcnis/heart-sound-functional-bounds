"""Independent re-inference and arithmetic audit of all adaptive follow-up arms."""
from pathlib import Path
import numpy as np
import torch
from query_training_common import *
from audit_query_pilot import recompute,independent_metrics
from train_query_pilot import state_hash
from train_query_warmstart import OUT,CFG,PROTO

def nominal_reference(p,graphs):
    T=len(p);result=[]
    for pm,d,a in graphs(p):
        path=solve(*prepare(pm,d,a))[3];r=np.asarray(pm,dtype=np.float64)[np.arange(T),path]
        result.append([r.sum()/T,r[:T//2].sum()/T,r[T//2:].sum()/T])
    return np.asarray(result)

def main():
    path=OUT/'independent_audit.json'
    if path.exists():raise FileExistsError(path)
    run=load(OUT/'run.json');assert run['status']=='complete';verify_hashes(run['sources'])
    assert digest(OUT/'summary.json')==run['summary_sha256']
    assert digest(OUT/'teacher_nominal_targets.json')==run['teacher_nominal_targets_sha256']
    for rel,sha in run['arm_run_sha256'].items():assert digest(OUT/rel)==sha
    cfg=load(CFG);_,rows=read_cache();dev=[r for r in rows if r['role']=='development'];graphs=Potentials()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    targets=load(OUT/'teacher_nominal_targets.json');target_error=0.;teacher={}
    for row,reported in zip(rows,targets):
        assert row['record']==reported['record']
        c=nominal_reference(row['teacher_T1'],graphs);teacher[row['record']]=c
        target_error=max(target_error,float(np.abs(c-np.asarray(reported['nominal'])).max()))
    forward_error=0.;query_error=0.;metric_error=0.;response_error=0.;checks=0;values=[]
    for seed in cfg['seeds']:
        initializations=[];orders=[]
        for arm in cfg['arms']:
            directory=OUT/f'seed_{seed}_{arm}';info=load(directory/'run.json')
            assert info['status']=='complete' and info['sources']==run['sources']
            for name,sha in info['output_sha256'].items():assert digest(directory/name)==sha
            initializations.append(info['initialization_sha256']);orders.append(info['order_sha256'])
            assert info['initialization_sha256']==load(RESULTS/f'seed_{seed}_kl/run.json')['final_state_sha256']
            expected=[np.random.default_rng(seed+(20+e)*1000003).permutation(80).tolist() for e in range(20)]
            assert load(directory/'orders.json')==expected
            history=load(directory/'history.json');assert len(history)==20
            assert all(np.isfinite(v) for h in history for v in h.values())
            checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
            assert checkpoint['total_epoch']==40 and checkpoint['config']==cfg
            model=PhysiologicalStudent();model.load_state_dict(checkpoint['model_state_dict']);model.to('cuda').eval()
            assert state_hash(model)==info['final_state_sha256']
            reported=load(directory/'development_metrics.json');responses=load(directory/'response_metrics.json')
            assert len(reported)==len(responses)==24;independent=[]
            for row,r,rr in zip(dev,reported,responses):
                assert r['record']==rr['record']==row['record'] and r['group_id']==row['group_id']
                with np.load(directory/r['array'],allow_pickle=False) as z:
                    p=z['posterior'].copy();p2=z['posterior_T2'].copy();savedg=z['gaps'].copy();saveds=float(z['score'])
                with torch.inference_mode():
                    x,l=batch_features([row],'cuda');logits=model(x,l)[0].T
                    for actual,wanted in [(logits.softmax(-1).cpu().numpy(),p),((logits/2.).softmax(-1).cpu().numpy(),p2)]:
                        forward_error=max(forward_error,float(np.abs(actual-wanted).max()))
                g,s=recompute(p,graphs);c=nominal_reference(p,graphs);tc=teacher[row['record']]
                query_error=max(query_error,float(np.abs(g-savedg).max()),abs(s-saveds))
                actual=independent_metrics(row['teacher_T1'],row['teacher_T2'],p2,g,row['teacher_gaps'],s,float(row['teacher_score']),cfg['eta'])
                for key,v in actual.items():metric_error=max(metric_error,abs(v-r[key]))
                response=float(np.abs((g-c[:,:,None])-(row['teacher_gaps']-tc[:,:,None])).mean())
                response_error=max(response_error,abs(response-rr['response_mae']),float(np.abs(c-np.asarray(rr['nominal'])).max()))
                actual['response_mae']=response;independent.append(actual);checks+=1
                for eta in cfg['sign_etas']:
                    signs=[]
                    for v in [g,row['teacher_gaps']]:
                        lo=v[1:,0,0].max()-v[0,0,1]-2*eta;hi=v[1:,0,1].max()-v[0,0,0]+2*eta
                        signs.append('positive' if lo>0 else 'nonpositive' if hi<=0 else 'unresolved')
                    assert r['sign_agreement'][str(eta)]==dict(student=signs[0],teacher=signs[1],agrees=signs[0]==signs[1])
            average={k:float(np.mean([r[k] for r in independent])) for k in independent[0]}
            for k,v in average.items():metric_error=max(metric_error,abs(v-info['metrics'][k]))
            values.append(dict(seed=seed,arm=arm,metrics=average))
        assert len(set(initializations))==len(set(orders))==1
    assert target_error<1e-12 and response_error<1e-12 and query_error<1e-11 and metric_error<5e-7 and forward_error<=1e-7
    means={a:{k:float(np.mean([r['metrics'][k] for r in values if r['arm']==a])) for k in values[0]['metrics']} for a in cfg['arms']}
    controls=['kl','score','confidence'];improvement={a:1-means['query']['primary_bound_mae']/means[a]['primary_bound_mae'] for a in controls}
    response_gain=1-means['query']['response_mae']/means['confidence']['response_mae']
    perseed={seed:{r['arm']:r['metrics']['primary_bound_mae'] for r in values if r['seed']==seed} for seed in cfg['seeds']}
    wins=sum(all(perseed[s]['query']<perseed[s][a] for a in controls) for s in cfg['seeds'])
    ratio=means['query']['emission_kl']/means['kl']['emission_kl']
    gate=all(v>=.1 for v in improvement.values()) and wins>=2 and ratio<=1.05 and response_gain>=.1
    summary=load(OUT/'summary.json');assert gate==summary['engineering_gate_passed']
    for a in cfg['arms']:
        for k,v in means[a].items():assert abs(v-summary['means'][a][k])<5e-7
    write(path,dict(status='passed',models=12,student_records=checks,teacher_nominal_records=len(rows),
        max_teacher_nominal_error=target_error,max_student_forward_error=forward_error,max_query_error=query_error,
        max_metric_error=metric_error,max_response_error=response_error,means=means,engineering_gate_passed=gate,
        improvement=improvement,response_gain=response_gain,paired_seed_wins=wins,emission_ratio=ratio,
        sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),ROOT/'phase10/src/audit_query_pilot.py',
                    OUT/'run.json',OUT/'summary.json']},scope='All saved-posterior reinference and independent arithmetic; same exhaustive-tested DP oracle'))
    print('AUDIT_PASSED '+str(dict(records=checks,gate=gate,max_metric_error=metric_error)))

if __name__=='__main__':main()
