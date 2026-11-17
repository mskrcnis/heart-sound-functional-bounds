"""Frozen adaptive follow-up separating support response from nominal confidence."""
import os
import random
import time
import torch.nn.functional as F
from query_training_common import *
from query_nominal_control import nominal_numpy,nominal_torch
from train_query_pilot import state_hash,monitor_loss,evaluate

OUT=ROOT/'phase10/results/query_gap_warmstart_v1'
CFG=ROOT/'phase10/configs/query_gap_warmstart_v1.json'
PROTO=ROOT/'phase10/QUERY_GAP_WARMSTART_PROTOCOL_v1.txt'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();started=time.time();cfg=load(CFG);cache,rows=read_cache()
    prior=load(RESULTS/'run.json');audit=load(RESULTS/'independent_audit.json')
    assert prior['status']=='complete' and audit['status']=='passed' and not audit['engineering_gate_passed']
    verify_hashes(prior['sources'])
    sources=dict(prior['sources'])
    for p in [CFG,PROTO,Path(__file__).resolve(),ROOT/'phase10/src/query_nominal_control.py',
              RESULTS/'run.json',RESULTS/'summary.json',RESULTS/'independent_audit.json']:
        sources[str(p.relative_to(ROOT))]=digest(p)
    for seed in cfg['seeds']:
        directory=RESULTS/f'seed_{seed}_kl';r=load(directory/'run.json')
        for name in ['final.pt','run.json']:
            p=directory/name;sources[str(p.relative_to(ROOT))]=digest(p)
        assert digest(directory/'final.pt')==r['output_sha256']['final.pt']
    write(OUT/'run.json',dict(status='sealed_before_additional_fitting',sources=sources,config=cfg,started_unix=started,pid=os.getpid()))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device='cuda';potentials=Potentials();checks=[]
    for row in rows:
        row['teacher_nominal']=nominal_numpy(row['teacher_T1'],potentials)
        checks.append(dict(record=row['record'],nominal=row['teacher_nominal'].tolist()))
    # Value and gradient checks on each role before any optimizer step.
    for role in ['fit','monitor','development']:
        row=next(r for r in rows if r['role']==role)
        x=torch.tensor(row['teacher_T1'],device=device,requires_grad=True)
        value=nominal_torch(x,potentials)
        np.testing.assert_allclose(value.detach().cpu().numpy(),row['teacher_nominal'],atol=2e-13,rtol=0.)
        value.sum().backward();assert torch.isfinite(x.grad).all() and float(x.grad.abs().sum())>0
    write(OUT/'teacher_nominal_targets.json',checks)
    fit=[r for r in rows if r['role']=='fit'];monitor=[r for r in rows if r['role']=='monitor'];dev=[r for r in rows if r['role']=='development']
    summaries=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            directory=OUT/f'seed_{seed}_{arm}';directory.mkdir();begin=time.time()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=PhysiologicalStudent().to(device)
            checkpoint=torch.load(RESULTS/f'seed_{seed}_kl/final.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict']);initial=state_hash(model)
            assert initial==load(RESULTS/f'seed_{seed}_kl/run.json')['final_state_sha256']
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);orders=[];history=[]
            for epoch in range(cfg['epochs']):
                tick=time.time();model.train();totals=np.zeros(3);n=0
                order=np.random.default_rng(seed+(20+epoch)*1000003).permutation(len(fit)).tolist();orders.append(order)
                for i in range(0,len(fit),cfg['batch_size']):
                    batch=[fit[j] for j in order[i:i+cfg['batch_size']]]
                    x,lengths=batch_features(batch,device);optimizer.zero_grad(set_to_none=True)
                    logits=model(x,lengths);kl=emission_loss(logits,batch,lengths);aux=kl.new_zeros(())
                    if arm!='kl':
                        losses=[]
                        for j,(row,T) in enumerate(zip(batch,lengths.tolist())):
                            p=logits[j,:,:T].T.softmax(-1)
                            if arm=='confidence':value=nominal_torch(p,potentials);teacher=row['teacher_nominal']
                            else:
                                value=targets_torch(p,potentials,arm)
                                teacher=row['teacher_gaps'] if arm=='query' else row['teacher_score']
                            target=torch.as_tensor(teacher,dtype=torch.float64,device=device)
                            losses.append(F.smooth_l1_loss(value,target,beta=cfg['smooth_l1_beta']))
                        aux=torch.stack(losses).mean()
                    loss=kl+cfg['auxiliary_weight']*aux
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite/missing gradient')
                    optimizer.step();totals+=np.array([float(loss.detach()),float(kl.detach()),float(aux.detach())])*len(batch);n+=len(batch)
                mon=monitor_loss(model,monitor,device)
                h=dict(epoch=epoch+1,total_epoch=epoch+21,loss=totals[0]/n,emission_kl=totals[1]/n,auxiliary=totals[2]/n,monitor_emission_kl=mon,seconds=time.time()-tick)
                history.append(h);write(directory/'history.json',history)
                print(f'{seed} {arm} warm_epoch={epoch+1:02d} KL={h["emission_kl"]:.6f} monitorKL={mon:.6f} sec={h["seconds"]:.2f}',flush=True)
            torch.save(dict(model_state_dict=model.state_dict(),seed=seed,arm=arm,epoch=20,total_epoch=40,config=cfg),directory/'final.pt')
            write(directory/'orders.json',orders)
            metric=evaluate(model,dev,potentials,directory,device,cfg)
            response=[]
            for i,row in enumerate(dev):
                with np.load(directory/f'development_{i:02d}.npz',allow_pickle=False) as z:p=z['posterior'].copy();g=z['gaps'].copy()
                c=nominal_numpy(p,potentials);teacher_response=row['teacher_gaps']-row['teacher_nominal'][:,:,None]
                mae=float(np.mean(np.abs((g-c[:,:,None])-teacher_response)))
                response.append(dict(record=row['record'],group_id=row['group_id'],response_mae=mae,nominal=c.tolist()))
            write(directory/'response_metrics.json',response);metric['response_mae']=float(np.mean([r['response_mae'] for r in response]))
            verify_hashes(sources)
            result=dict(status='complete',seed=seed,arm=arm,sources=sources,config=cfg,initialization_sha256=initial,
                order_sha256=digest(directory/'orders.json'),final_state_sha256=state_hash(model),seconds=time.time()-begin,
                metrics=metric,output_sha256={p.name:digest(p) for p in directory.iterdir()})
            write(directory/'run.json',result);summaries.append(result)
            print('ARM_COMPLETE '+json.dumps(dict(seed=seed,arm=arm,metrics=metric)),flush=True)
    keys=['emission_kl','score_mae','global_bound_mae','window_bound_mae','primary_bound_mae','support_gap_mae','response_mae']
    means={a:{k:float(np.mean([r['metrics'][k] for r in summaries if r['arm']==a])) for k in keys} for a in cfg['arms']}
    perseed={seed:{r['arm']:r['metrics']['primary_bound_mae'] for r in summaries if r['seed']==seed} for seed in cfg['seeds']}
    for seed in cfg['seeds']:
        sub=[r for r in summaries if r['seed']==seed]
        assert len({r['initialization_sha256'] for r in sub})==1 and len({r['order_sha256'] for r in sub})==1
    controls=['kl','score','confidence'];improve={a:1-means['query']['primary_bound_mae']/means[a]['primary_bound_mae'] for a in controls}
    response_gain=1-means['query']['response_mae']/means['confidence']['response_mae']
    wins=sum(all(perseed[s]['query']<perseed[s][a] for a in controls) for s in cfg['seeds'])
    klratio=means['query']['emission_kl']/means['kl']['emission_kl']
    gate=all(v>=.1 for v in improve.values()) and wins>=2 and klratio<=1.05 and response_gain>=.1
    summary=dict(status='complete',means=means,per_seed=perseed,relative_primary_improvement=improve,
        relative_response_improvement_vs_confidence=response_gain,paired_seed_wins=wins,emission_kl_ratio=klratio,
        engineering_gate_passed=gate,adaptive=True,n=24,all_arms=[dict(seed=r['seed'],arm=r['arm'],metrics=r['metrics'],seconds=r['seconds']) for r in summaries])
    write(OUT/'summary.json',summary);verify_hashes(sources)
    write(OUT/'run.json',dict(status='complete',config=cfg,sources=sources,completed_models=12,started_unix=started,seconds=time.time()-started,
        summary_sha256=digest(OUT/'summary.json'),arm_run_sha256={str((OUT/f'seed_{r["seed"]}_{r["arm"]}'/'run.json').relative_to(OUT)):
          digest(OUT/f'seed_{r["seed"]}_{r["arm"]}'/'run.json') for r in summaries},
        teacher_nominal_targets_sha256=digest(OUT/'teacher_nominal_targets.json')))
    print('FOLLOWUP_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
