"""Execute all frozen phase10 pilot arms; completed directories are verified."""
import os
import platform
import random
import time
import torch.nn.functional as F
from query_training_common import *

def state_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def monitor_loss(model,rows,device):
    values=[];model.eval()
    with torch.inference_mode():
        for i in range(0,len(rows),8):
            batch=rows[i:i+8];x,lengths=batch_features(batch,device)
            values.append(float(emission_loss(model(x,lengths),batch,lengths))*len(batch))
    return sum(values)/len(rows)

def evaluate(model,rows,potentials,out,device,cfg):
    model.eval();results=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            x,lengths=batch_features([row],device);logits=model(x,lengths)[0].T
            p=logits.softmax(-1).cpu().numpy();p2=(logits/2.).softmax(-1).cpu().numpy()
            g,s=targets_numpy(p,potentials)
            array=f'development_{i:02d}.npz'
            np.savez_compressed(out/array,posterior=p,posterior_T2=p2,gaps=g,score=np.asarray(s))
            result=metrics(p2,row['teacher_T2'],g,s,row['teacher_gaps'],row['teacher_score'],cfg['eta'],cfg['sign_etas'])
            results.append(dict(record=row['record'],group_id=row['group_id'],array=array,
                                sha256=digest(out/array),**result))
    write(out/'development_metrics.json',results)
    keys=['emission_kl','score_mae','global_bound_mae','window_bound_mae','primary_bound_mae','support_gap_mae']
    return dict({k:float(np.mean([r[k] for r in results])) for k in keys},
                sign_agreement={str(e):sum(r['sign_agreement'][str(e)]['agrees'] for r in results)/len(results) for e in cfg['sign_etas']},
                n=len(results))

def main():
    started=time.time();cfg=load(CONFIG);cache,rows=read_cache()
    fit=[r for r in rows if r['role']=='fit'];monitor=[r for r in rows if r['role']=='monitor']
    dev=[r for r in rows if r['role']=='development']
    assert (len(fit),len(monitor),len(dev))==(80,16,24)
    sources=dict(cache['sources'])
    for p in [Path(__file__).resolve(),CACHE/'manifest.json',ROOT/'phase10/tests/test_query_training.py',
              ROOT/'phase10/tests/test_query_gap_distillation.py']:
        sources[str(p.relative_to(ROOT))]=digest(p)
    RESULTS.mkdir(exist_ok=True)
    if (RESULTS/'run.json').exists():
        previous=load(RESULTS/'run.json');assert previous['sources']==sources and previous['config']==cfg
        if previous['status']=='complete':raise FileExistsError('Pilot already complete; do not rerun')
    write(RESULTS/'run.json',dict(status='running',config=cfg,sources=sources,started_unix=started,
        pid=os.getpid(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;device='cuda';potentials=Potentials();summaries=[]
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            out=RESULTS/f'seed_{seed}_{arm}'
            if out.exists():
                done=load(out/'run.json')
                if done['status']!='complete' or done['sources']!=sources:raise ValueError('Incomplete or mismatched prior arm')
                for filename,sha in done['output_sha256'].items():assert digest(out/filename)==sha
                summaries.append(done);continue
            out.mkdir();begin=time.time()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=PhysiologicalStudent().to(device);initialization_sha=state_hash(model)
            assert sum(p.numel() for p in model.parameters())==4877
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
            history=[];orders=[]
            write(out/'run.json',dict(status='running',seed=seed,arm=arm,sources=sources,initialization_sha256=initialization_sha))
            for epoch in range(cfg['epochs']):
                epoch_begin=time.time();model.train()
                order=np.random.default_rng(seed+epoch*1000003).permutation(len(fit)).tolist();orders.append(order)
                sums=np.zeros(3);count=0
                for i in range(0,len(order),cfg['batch_size']):
                    batch=[fit[j] for j in order[i:i+cfg['batch_size']]]
                    x,lengths=batch_features(batch,device);optimizer.zero_grad(set_to_none=True)
                    logits=model(x,lengths);kl=emission_loss(logits,batch,lengths);aux=kl.new_zeros(())
                    if arm!='kl':
                        losses=[]
                        for j,(row,T) in enumerate(zip(batch,lengths.tolist())):
                            p=logits[j,:,:T].T.softmax(-1)
                            value=targets_torch(p,potentials,arm)
                            teacher=row['teacher_gaps'] if arm=='query' else row['teacher_score']
                            target=torch.as_tensor(teacher,dtype=torch.float64,device=device)
                            losses.append(F.smooth_l1_loss(value,target,beta=cfg['smooth_l1_beta']))
                        aux=torch.stack(losses).mean()
                    loss=kl+cfg['auxiliary_weight']*aux
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('Missing/nonfinite gradient')
                    optimizer.step()
                    sums+=np.asarray([float(loss.detach()),float(kl.detach()),float(aux.detach())])*len(batch);count+=len(batch)
                mon=monitor_loss(model,monitor,device)
                row=dict(epoch=epoch+1,loss=sums[0]/count,emission_kl=sums[1]/count,auxiliary=sums[2]/count,
                         monitor_emission_kl=mon,seconds=time.time()-epoch_begin)
                history.append(row);write(out/'history.json',history)
                print(f'{seed} {arm} epoch={epoch+1:02d} loss={row["loss"]:.6f} KL={row["emission_kl"]:.6f} monitorKL={mon:.6f} sec={row["seconds"]:.2f}',flush=True)
            torch.save(dict(model_state_dict=model.state_dict(),seed=seed,arm=arm,epoch=cfg['epochs'],
                       initialization_sha256=initialization_sha,config=cfg),out/'final.pt')
            write(out/'orders.json',orders)
            metric=evaluate(model,dev,potentials,out,device,cfg)
            verify_hashes(sources)
            result=dict(status='complete',seed=seed,arm=arm,config=cfg,sources=sources,
                initialization_sha256=initialization_sha,order_sha256=digest(out/'orders.json'),
                final_state_sha256=state_hash(model),seconds=time.time()-begin,metrics=metric,
                parameter_count=4877,output_sha256={p.name:digest(p) for p in out.iterdir() if p.name!='run.json'})
            write(out/'run.json',result);summaries.append(result)
            print('ARM_COMPLETE '+json.dumps(dict(seed=seed,arm=arm,metrics=metric)),flush=True)
    for seed in cfg['seeds']:
        sub=[r for r in summaries if r['seed']==seed]
        assert len({r['initialization_sha256'] for r in sub})==1
        assert len({r['order_sha256'] for r in sub})==1
    means={arm:{k:float(np.mean([r['metrics'][k] for r in summaries if r['arm']==arm])) for k in
        ['emission_kl','score_mae','global_bound_mae','window_bound_mae','primary_bound_mae','support_gap_mae']} for arm in cfg['arms']}
    per_seed={seed:{r['arm']:r['metrics']['primary_bound_mae'] for r in summaries if r['seed']==seed} for seed in cfg['seeds']}
    improvement={arm:1.-means['query']['primary_bound_mae']/means[arm]['primary_bound_mae'] for arm in ['kl','score']}
    wins=sum(all(per_seed[s]['query']<per_seed[s][a] for a in ['kl','score']) for s in cfg['seeds'])
    emission_ratio=means['query']['emission_kl']/means['kl']['emission_kl']
    gate=all(v>=.1 for v in improvement.values()) and wins>=2 and emission_ratio<=1.05
    write(RESULTS/'summary.json',dict(status='complete',means=means,per_seed=per_seed,
        relative_primary_improvement=improvement,paired_seeds_better_than_both=wins,
        query_to_kl_emission_error_ratio=emission_ratio,engineering_gate_passed=gate,
        interpretation='Adaptive teacher-fidelity pilot; no clinical labels or independent confirmation',
        all_arms=[dict(seed=r['seed'],arm=r['arm'],metrics=r['metrics'],seconds=r['seconds']) for r in summaries]))
    verify_hashes(sources)
    write(RESULTS/'run.json',dict(status='complete',config=cfg,sources=sources,started_unix=started,
         elapsed_seconds=time.time()-started,completed_models=len(summaries),pid=os.getpid(),
         python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
         summary_sha256=digest(RESULTS/'summary.json'),arm_run_sha256={str((RESULTS/f'seed_{r["seed"]}_{r["arm"]}'/'run.json').relative_to(RESULTS)):
             digest(RESULTS/f'seed_{r["seed"]}_{r["arm"]}'/'run.json') for r in summaries}))
    print('PILOT_COMPLETE '+json.dumps(load(RESULTS/'summary.json')),flush=True)

if __name__=='__main__':main()
