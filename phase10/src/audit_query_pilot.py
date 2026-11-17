"""Recompute saved-posteriors and metrics independently of the training reporter."""
from query_training_common import (ROOT,P9,CACHE,RESULTS,CONFIG,PhysiologicalStudent,
    Potentials,prepare,solve,load,write,digest,verify_hashes,read_cache,import_path,batch_features)
import hashlib
import numpy as np
import torch

def recompute(p,graph):
    T=len(p);out=np.zeros((4,3,2));conf=[]
    for m,(emission,duration,transition) in enumerate(graph(p)):
        args=prepare(emission,duration,transition);zero=solve(*args)
        conf.append(np.asarray(emission,dtype=np.float64)[np.arange(T),zero[3]].mean())
        for k in range(3):
            weighted=np.asarray(emission,dtype=np.float64).copy()
            if k==1:weighted[T//2:]=0.
            if k==2:weighted[:T//2]=0.
            reward_prefix=np.concatenate((np.zeros((1,emission.shape[1])),weighted.cumsum(axis=0)))
            for j,lam in enumerate((-1.,1.)):
                value=solve(args[0],reward_prefix,args[2],args[3],args[4],lam=lam)[0]
                out[m,k,j]=(value-zero[0])/(T*lam)
    return out,max(conf[1:])-conf[0]

def independent_metrics(q,q2,s2,g,t,score,ts,eta):
    lo=g[:,:,0]-eta;hi=g[:,:,1]+eta;tl=t[:,:,0]-eta;th=t[:,:,1]+eta
    low=max(lo[1:,0])-hi[0,0];high=max(hi[1:,0])-lo[0,0]
    tlow=max(tl[1:,0])-th[0,0];thigh=max(th[1:,0])-tl[0,0]
    global_error=(abs(low-tlow)+abs(high-thigh))/2
    window_error=(np.abs(lo[:,1:]-tl[:,1:]).sum()+np.abs(hi[:,1:]-th[:,1:]).sum())/16
    kl=(q2.astype('float64')*(np.log(q2.astype('float64'))-np.log(s2.astype('float64')))).sum(axis=1).mean()*4
    return {'emission_kl':float(kl),'score_mae':abs(score-ts),'global_bound_mae':global_error,
            'window_bound_mae':window_error,'primary_bound_mae':(global_error+window_error)/2,
            'support_gap_mae':float(np.abs(g-t).mean())}

def main():
    out=RESULTS/'independent_audit.json'
    if out.exists():raise FileExistsError(out)
    run=load(RESULTS/'run.json');assert run['status']=='complete'
    verify_hashes(run['sources']);cfg=load(CONFIG)
    assert digest(RESULTS/'summary.json')==run['summary_sha256']
    for rel,sha in run['arm_run_sha256'].items():assert digest(RESULTS/rel)==sha
    cache,rows=read_cache();dev=[r for r in rows if r['role']=='development']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    graph=Potentials();teacher_delta=0.;query_delta=0.;metric_delta=0.;forward_delta=0.;checks=0
    for row in rows:
        g,s=recompute(row['teacher_T1'],graph)
        teacher_delta=max(teacher_delta,float(np.abs(g-row['teacher_gaps']).max()),abs(s-float(row['teacher_score'])))
    assert teacher_delta<1e-11
    # Re-infer one cached feature sequence per role with all five teachers.
    neural=import_path('phase10_audit_neural',P9/'third_party/parallel-hsmm-murmur/src/neural_networks.py')
    base=load(P9/'configs/hsmm_baseline.json');models=[]
    for fold in range(5):
        c=torch.load(P9/'results/hsmm_baseline_v1/models'/f'fold_{fold}.pt',map_location='cpu',weights_only=False)
        model=neural.instantiate_model(base['architecture']);model.load_state_dict(c['model_state_dict']);model.to('cuda').eval();models.append(model)
    teacher_forward_delta=0.
    with torch.inference_mode():
        for role in ['fit','monitor','development']:
            row=next(r for r in rows if r['role']==role);x,lengths=batch_features([row],'cuda')
            logits=[m(x,lengths)[0] for m in models]
            for temp,key in [(1.,'teacher_T1'),(2.,'teacher_T2')]:
                value=torch.stack([(v/temp).softmax(0) for v in logits]).mean(0).T.cpu().numpy()
                teacher_forward_delta=max(teacher_forward_delta,float(np.abs(value-row[key]).max()))
    assert teacher_forward_delta<=1e-7
    del models
    audit_rows=[];means={}
    for seed in cfg['seeds']:
        initializations=[];orders=[]
        for arm in cfg['arms']:
            directory=RESULTS/f'seed_{seed}_{arm}';record=load(directory/'run.json')
            assert record['status']=='complete' and record['sources']==run['sources']
            for name,sha in record['output_sha256'].items():assert digest(directory/name)==sha
            initializations.append(record['initialization_sha256']);orders.append(record['order_sha256'])
            torch.manual_seed(seed);fresh=PhysiologicalStudent();h=hashlib.sha256()
            for name,value in sorted(fresh.state_dict().items()):h.update(name.encode());h.update(value.numpy().tobytes())
            assert h.hexdigest()==record['initialization_sha256']
            history=load(directory/'history.json');assert len(history)==cfg['epochs']
            assert all(np.isfinite(v) for r in history for v in r.values())
            c=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
            assert c['epoch']==cfg['epochs'] and c['seed']==seed and c['arm']==arm
            fresh.load_state_dict(c['model_state_dict']);fresh.to('cuda').eval()
            reported=load(directory/'development_metrics.json');assert len(reported)==24
            independent=[]
            for row,r in zip(dev,reported):
                assert r['record']==row['record'] and r['group_id']==row['group_id']
                assert digest(directory/r['array'])==r['sha256']
                with np.load(directory/r['array'],allow_pickle=False) as z:
                    p=z['posterior'].copy();p2=z['posterior_T2'].copy();g0=z['gaps'].copy();s0=float(z['score'])
                assert np.isfinite(p).all() and (p>0).all()
                with torch.inference_mode():
                    x,lengths=batch_features([row],'cuda');logit=fresh(x,lengths)[0].T
                    for value,expected in [(logit.softmax(-1).cpu().numpy(),p),((logit/2.).softmax(-1).cpu().numpy(),p2)]:
                        forward_delta=max(forward_delta,float(np.abs(value-expected).max()))
                g,s=recompute(p,graph)
                query_delta=max(query_delta,float(np.abs(g-g0).max()),abs(s-s0))
                m=independent_metrics(row['teacher_T1'],row['teacher_T2'],p2,g,row['teacher_gaps'],s,float(row['teacher_score']),cfg['eta'])
                for key,value in m.items():metric_delta=max(metric_delta,abs(value-r[key]))
                for eta in cfg['sign_etas']:
                    sign=[]
                    for values in [g,row['teacher_gaps']]:
                        low=max(values[1:,0,0]-eta)-(values[0,0,1]+eta)
                        high=max(values[1:,0,1]+eta)-(values[0,0,0]-eta)
                        sign.append('positive' if low>0 else 'nonpositive' if high<=0 else 'unresolved')
                    assert r['sign_agreement'][str(eta)]==dict(student=sign[0],teacher=sign[1],agrees=sign[0]==sign[1])
                independent.append(m);checks+=1
            average={k:float(np.mean([r[k] for r in independent])) for k in independent[0]}
            for key,value in average.items():metric_delta=max(metric_delta,abs(value-record['metrics'][key]))
            audit_rows.append(dict(seed=seed,arm=arm,metrics=average))
        assert len(set(initializations))==1 and len(set(orders))==1
    assert query_delta<1e-11 and metric_delta<5e-7 and forward_delta<=1e-7
    means={arm:{key:float(np.mean([r['metrics'][key] for r in audit_rows if r['arm']==arm]))
                for key in audit_rows[0]['metrics']} for arm in cfg['arms']}
    summary=load(RESULTS/'summary.json')
    for arm in cfg['arms']:
        for key,value in means[arm].items():assert abs(value-summary['means'][arm][key])<5e-7
    improvements={a:1-means['query']['primary_bound_mae']/means[a]['primary_bound_mae'] for a in ['kl','score']}
    byseed={s:{r['arm']:r['metrics']['primary_bound_mae'] for r in audit_rows if r['seed']==s} for s in cfg['seeds']}
    wins=sum(all(byseed[s]['query']<byseed[s][a] for a in ['kl','score']) for s in cfg['seeds'])
    ratio=means['query']['emission_kl']/means['kl']['emission_kl']
    gate=all(v>=.1 for v in improvements.values()) and wins>=2 and ratio<=1.05
    assert gate==summary['engineering_gate_passed']
    write(out,dict(status='passed',teacher_target_records_checked=len(rows),teacher_forward_records_checked=3,
        final_student_records_checked=checks,final_models_checked=9,teacher_target_max_error=teacher_delta,
        teacher_forward_max_error=teacher_forward_delta,student_forward_max_error=forward_delta,
        query_max_error=query_delta,metric_max_error=metric_delta,means=means,
        engineering_gate_passed=gate,relative_primary_improvement=improvements,paired_seed_wins=wins,
        query_to_kl_emission_ratio=ratio,sources={str(p.relative_to(ROOT)):digest(p) for p in
            [Path(__file__).resolve(),RESULTS/'run.json',RESULTS/'summary.json',CACHE/'manifest.json']},
        scope='Independent metric aggregation and saved-posterior re-inference; same already exhaustive-tested DP oracle'))
    print('AUDIT_PASSED '+str(dict(records=checks,max_metric_error=metric_delta,gate=gate)))

if __name__=='__main__':
    from pathlib import Path
    main()
