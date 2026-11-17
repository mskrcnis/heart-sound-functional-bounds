"""Seal label-free selection, then compute continuous-prefix teacher targets."""
import platform
import time
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from query_training_common import *

def main():
    if CACHE.exists():raise FileExistsError(CACHE)
    CACHE.mkdir(parents=True);start=time.time()
    cfg=load(CONFIG);base=load(P9/'configs/hsmm_baseline.json')
    metadata=load(P9/'results/metadata/manifest.json')
    for rel,sha in metadata['output_sha256'].items():
        assert digest(P9/'results/metadata'/rel)==sha
    invpath=P9/'results/metadata/record_inventory.csv'
    inv=pd.read_csv(invpath,usecols=['patient_id','site','record','selected','group_id','role'],dtype=str)
    inv=inv[inv.selected.eq('True') & inv.role.eq('train') & inv.site.isin(['AV','PV','TV','MV'])]
    candidates={}
    for row in inv.to_dict('records'):candidates.setdefault(row['group_id'],[]).append(row)
    groups=sorted(candidates,key=lambda g:hashlib.sha256((cfg['selection_prefix']+g).encode()).hexdigest())
    selected=[]
    for i,g in enumerate(groups[:cfg['fit_people']+cfg['monitor_people']]):
        r=sorted(candidates[g],key=lambda r:(['AV','PV','TV','MV'].index(r['site']),r['record']))[0]
        selected.append(dict(record=r['record'],patient_id=r['patient_id'],group_id=g,
                             source_role='train',role='fit' if i<cfg['fit_people'] else 'monitor'))
    oldselection=ROOT/'phase10/results/development_inputs_v1/selection_manifest.json'
    for r in load(oldselection)['records']:
        selected.append(dict(record=r['record'],patient_id=r['patient_id'],group_id=r['group_id'],
                             source_role='pilot_validation',role='development'))
    if len(selected)!=120 or len({r['group_id'] for r in selected})!=120:raise AssertionError('Selection/group overlap')
    splitpath=P9/'results/metadata/split_manifest.csv'
    split=pd.read_csv(splitpath,usecols=['Patient ID','group_id','role'],dtype=str).set_index('Patient ID')
    for r in selected:
        s=split.loc[r['patient_id']]
        assert s.group_id==r['group_id'] and s.role==r['source_role']
    checks_path=P9/'data/circor_metadata/SHA256SUMS.txt'
    checks=dict((f.lstrip('*'),h) for h,f in (line.split(maxsplit=1) for line in checks_path.read_text().splitlines()))
    paths=source_paths()+[Path(__file__).resolve(),invpath,splitpath,oldselection,checks_path,
                          P9/'results/metadata/manifest.json',P9/'results/hsmm_baseline_v1/run.json',
                          P9/'results/hsmm_training_cache_v1/records.csv',
                          P9/'results/hsmm_training_cache_v1/manifest.json']
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    teachers_cfg=load(P9/'configs/physiological_distill.json')
    for name,sha in teachers_cfg['teacher_checkpoint_sha256'].items():
        p=P9/'results/hsmm_baseline_v1/models'/name
        assert digest(p)==sha;source_hashes[str(p.relative_to(ROOT))]=sha
    for r in selected:
        r['wav_sha256']=checks[r['record']+'.wav']
    write(CACHE/'selection.json',dict(status='sealed_before_waveform_reads',config=cfg,
        records=selected,sources=source_hashes,selection_used_labels=False,annotation_reads=0,
        final_reserve_inputs=0))
    selection_sha=digest(CACHE/'selection.json')
    # Direct neural import avoids compiling the upstream Cython decoder.
    run=load(P9/'results/hsmm_baseline_v1/run.json')
    assert run['status']=='complete' and run['config']==base and run['reserved_predictions']==0
    for rel,sha in run['input_sha256'].items():assert digest(P9/rel)==sha
    oldcache=P9/'results/hsmm_training_cache_v1'
    cm=load(oldcache/'manifest.json')
    assert digest(oldcache/'records.csv')==cm['files_sha256']['records.csv']
    membership=pd.read_csv(oldcache/'records.csv',usecols=['group_id','inner_fold','role'])
    assert membership.role.eq('train').all()
    neural=import_path('phase10_query_neural',P9/'third_party/parallel-hsmm-murmur/src/neural_networks.py')
    torch.set_num_threads(4);device='cuda';models=[]
    if not torch.cuda.is_available():raise RuntimeError('CUDA required by this execution')
    for fold in range(5):
        checkpoint=torch.load(P9/'results/hsmm_baseline_v1/models'/f'fold_{fold}.pt',map_location='cpu',weights_only=False)
        fit=set(membership.loc[membership.inner_fold.ne(fold),'group_id'])
        valid=set(membership.loc[membership.inner_fold.eq(fold),'group_id'])
        assert checkpoint['fold']==fold and checkpoint['architecture']==base['architecture']
        assert set(checkpoint['fit_group_ids'])==fit and set(checkpoint['inner_validation_group_ids'])==valid
        assert not fit & valid
        model=neural.instantiate_model(base['architecture'])
        model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.to(device).eval();models.append(model)
    potentials=Potentials();rows=[]
    for i,r in enumerate(selected):
        wav=P9/'data/development_wav'/(Path(r['record']).name+'.wav')
        assert digest(wav)==r['wav_sha256']
        fs,raw=wavfile.read(wav);raw=raw[:int(fs*cfg['prefix_seconds'])]
        features,detail=calculate_features(raw,fs,base)
        if features is None:raise ValueError((r['record'],detail))
        T=features.shape[1]
        with torch.inference_mode():
            x=features[None].to(device);lengths=torch.tensor([T],dtype=torch.long)
            logits=[m(x,lengths)[0] for m in models]
            t1=torch.stack([v.softmax(0) for v in logits]).mean(0).T.cpu().numpy()
            t2=torch.stack([(v/2.).softmax(0) for v in logits]).mean(0).T.cpu().numpy()
        gaps,score=targets_numpy(t1,potentials)
        array=f'{i:03d}.npz'
        np.savez_compressed(CACHE/array,features=features.numpy(),teacher_T1=t1,teacher_T2=t2,
                            teacher_gaps=gaps,teacher_score=np.asarray(score))
        rows.append(dict(r,array=array,sha256=digest(CACHE/array),frames=int(T),
                         prefix_seconds=len(raw)/fs,frontend_detail=detail))
        if (i+1)%20==0:print(f'cached {i+1}/{len(selected)}',flush=True)
    verify_hashes(source_hashes)
    write(CACHE/'manifest.json',dict(status='complete',selection_sha256=selection_sha,sources=source_hashes,
        records=rows,config=cfg,seconds=time.time()-start,python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda,teacher_same_fit=True,
        new_label_columns_read=False,new_annotations_read=0,final_reserve_inputs=0))
    print('COMPLETE: 80 fit, 16 monitor, 24 development continuous-prefix targets',flush=True)

if __name__=='__main__':main()
