# Public release adaptation notice, 2026-11-17: license notice added;
# numerical implementation unchanged. Derived boundary semantics: University
# of Cambridge, 2022. GAP Available Source License v1.0; see LICENSE.
"""Label-free continuous-input helpers for the frozen phase10 KD pilot.

Research derivative using Cambridge GAP ASL v1.0 code; see phase9/HSMM_LICENSE.md.
The upstream source is imported without compiling or changing its decoder.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import types
import numpy as np
import torch
from path_bounds import prepare, solve
from query_gap_distillation import query_gaps

ROOT=Path(__file__).resolve().parents[2]
P9=ROOT/'phase9'
CONFIG=ROOT/'phase10/configs/query_gap_pilot_v1.json'
PROTOCOL=ROOT/'phase10/QUERY_GAP_TRAINING_PROTOCOL_v1.txt'
CACHE=ROOT/'phase10/results/query_gap_inputs_v1'
RESULTS=ROOT/'phase10/results/query_gap_training_v1'
NAMES=['Healthy','Holosystolic','Early-systolic','Mid-systolic']
COLS=[[0,1,2,3],[0,4,2,3],[0,1,2,3,4],[0,1,2,3,4]]
sys.path.insert(0,str(P9/'src'))
sys.dont_write_bytecode=True
from physiological_distill import PhysiologicalStudent
from hsmm_baseline import calculate_features

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text())

def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class Potentials:
    def __init__(self):
        self.rows=[]
        capture=types.ModuleType('viterbi_hmm')
        def hsmm_viterbi(p,d,D,a):
            self.rows.append((p.copy(),d.copy(),a.copy()))
            return np.zeros(len(p),dtype=np.int32)
        capture.hsmm_viterbi=hsmm_viterbi
        cython=types.ModuleType('pyximport');cython.install=lambda:None
        old={n:sys.modules.get(n) for n in ['viterbi_hmm','pyximport']}
        try:
            sys.modules.update(viterbi_hmm=capture,pyximport=cython)
            self.module=import_path('phase10_query_segmenter',P9/'third_party/parallel-hsmm-murmur/src/segmenter.py')
        finally:
            for n,v in old.items():
                if v is None:sys.modules.pop(n,None)
                else:sys.modules[n]=v
        self.cfg=load(P9/'configs/hsmm_baseline.json')['hsmm']
    def __call__(self,p):
        self.rows.clear()
        self.module.double_duration_viterbi(np.asarray(p,dtype=np.float32),50,**self.cfg)
        if len(self.rows)!=4:raise AssertionError('All four topologies required')
        return list(self.rows)

def rewards(T):
    w=np.ones((3,T),dtype=np.float64)
    w[1,T//2:]=0.;w[2,:T//2]=0.
    return w

def targets_numpy(p,potentials):
    """Reference DP values, independent of torch selected-path reconstruction."""
    T=len(p);weights=rewards(T);gaps=np.empty((4,3,2));confidence=[]
    graphs=potentials(p)
    for m,(pm,d,a) in enumerate(graphs):
        args=prepare(pm,d,a);base=solve(*args)
        confidence.append(float(np.asarray(pm,dtype=np.float64)[np.arange(T),base[3]].mean()))
        for k,w in enumerate(weights):
            wrp=np.vstack((np.zeros(pm.shape[1]),np.cumsum(np.asarray(pm,dtype=np.float64)*w[:,None],axis=0)))
            pargs=(args[0],wrp,args[2],args[3],args[4])
            for j,lam in enumerate([-1.,1.]):
                gaps[m,k,j]=(solve(*pargs,lam=lam)[0]-base[0])/(lam*T)
    return gaps,float(max(confidence[1:])-confidence[0])

def targets_torch(p,potentials,kind):
    graphs=potentials(p.detach().cpu().numpy());T=len(p)
    if kind=='score':
        cs=[]
        for cols,(pm,d,a) in zip(COLS,graphs):
            path=solve(*prepare(pm,d,a))[3]
            idx=torch.as_tensor(path,dtype=torch.long,device=p.device)
            cs.append(p[:,cols][torch.arange(T,device=p.device),idx].double().mean())
        return torch.stack(cs[1:]).max()-cs[0]
    if kind!='query':raise ValueError(kind)
    weights=rewards(T);out=[]
    for cols,(pm,d,a) in zip(COLS,graphs):
        if not np.array_equal(pm,p.detach().cpu().numpy()[:,cols]):
            raise AssertionError('Unexpected topology emission mapping')
        out.append(torch.stack([query_gaps(p[:,cols],d,a,(-1.,1.),w) for w in weights]))
    return torch.stack(out)

def metrics(p,t2,gaps,score,teacher_gaps,teacher_score,eta=.005,sign_etas=(.001,.005,.01,.05)):
    # p is the T=2 student posterior for emission KL only.
    kl=float(np.mean(np.sum(t2*(np.log(t2)-np.log(p)),axis=1))*4.)
    def endpoints(g,e):
        lo=g[:,:,0]-e;hi=g[:,:,1]+e
        global_pair=np.array([lo[1:,0].max()-hi[0,0],hi[1:,0].max()-lo[0,0]])
        windows=np.stack((lo[:,1:],hi[:,1:]),axis=-1)
        return global_pair,windows
    gs,ws=endpoints(gaps,eta);gt,wt=endpoints(teacher_gaps,eta)
    ge=float(np.mean(np.abs(gs-gt)));we=float(np.mean(np.abs(ws-wt)))
    def sign(pair):return 'positive' if pair[0]>0 else 'nonpositive' if pair[1]<=0 else 'unresolved'
    agreement={}
    for e in sign_etas:
        student=sign(endpoints(gaps,e)[0]);teacher=sign(endpoints(teacher_gaps,e)[0])
        agreement[str(e)]={'student':student,'teacher':teacher,'agrees':student==teacher}
    return dict(emission_kl=kl,score_mae=abs(float(score)-float(teacher_score)),
                global_bound_mae=ge,window_bound_mae=we,primary_bound_mae=.5*(ge+we),
                support_gap_mae=float(np.mean(np.abs(gaps-teacher_gaps))),
                student_global_interval=gs.tolist(),teacher_global_interval=gt.tolist(),
                sign_agreement=agreement)

def batch_features(rows,device):
    lengths=torch.tensor([r['features'].shape[1] for r in rows],dtype=torch.long)
    x=torch.zeros(len(rows),40,int(lengths.max()),device=device)
    for i,r in enumerate(rows):x[i,:,:lengths[i]]=torch.as_tensor(r['features'],device=device)
    return x,lengths

def emission_loss(logits,rows,lengths):
    losses=[]
    for i,(row,T) in enumerate(zip(rows,lengths.tolist())):
        q=torch.as_tensor(row['teacher_T2'],device=logits.device)
        logp=torch.log_softmax(logits[i,:,:T].T/2.,dim=-1)
        losses.append((q*(q.log()-logp)).sum(-1).mean()*4.)
    return torch.stack(losses).mean()

def source_paths():
    return [CONFIG,PROTOCOL,Path(__file__).resolve(),
            ROOT/'phase10/src/query_gap_distillation.py',ROOT/'phase10/src/path_bounds.py',
            P9/'src/physiological_distill.py',P9/'src/hsmm_baseline.py',
            P9/'src/hsmm_traceback_sensitivity.py',P9/'src/hsmm_score_accounting.py',
            P9/'third_party/parallel-hsmm-murmur/src/segmenter.py',
            P9/'third_party/parallel-hsmm-murmur/src/neural_networks.py',
            P9/'configs/hsmm_baseline.json',P9/'configs/physiological_distill.json']

def verify_hashes(hashes):
    for relative,sha in hashes.items():
        if digest(ROOT/relative)!=sha:raise AssertionError('Changed source '+relative)

def read_cache():
    manifest=load(CACHE/'manifest.json')
    if manifest['status']!='complete':raise ValueError('Incomplete input cache')
    verify_hashes(manifest['sources'])
    if digest(CACHE/'selection.json')!=manifest['selection_sha256']:raise ValueError('Changed selection')
    result=[]
    for row in manifest['records']:
        if digest(CACHE/row['array'])!=row['sha256']:raise ValueError('Changed cache array')
        with np.load(CACHE/row['array'],allow_pickle=False) as z:
            result.append(dict(row,**{k:z[k].copy() for k in z.files}))
    return manifest,result
