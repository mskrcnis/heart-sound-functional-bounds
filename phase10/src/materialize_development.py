"""Select 24 groups without outcome columns, then capture fixed HSMM potentials.

Run with .venv-audio/bin/python. The isolated module uses the unchanged upstream
duration/topology construction; a recording decoder captures its inputs.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import types
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'phase10/results/development_inputs_v1'
EVAL=ROOT/'phase9/results/physiological_distill_v1/seed_20261107_alpha_1p0/evaluation'
SPLIT=ROOT/'phase9/results/metadata/split_manifest.csv'
SOURCE=ROOT/'phase9/third_party/parallel-hsmm-murmur/src/segmenter.py'
CFG=ROOT/'phase9/configs/hsmm_baseline.json'
PROTOCOL=ROOT/'phase10/PROTOCOL_v1.txt'
NAMES=['Healthy','Holosystolic','Early-systolic','Mid-systolic']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    meta=pd.read_csv(SPLIT,usecols=['Patient ID','group_id','role'],dtype=str)
    ids={r['Patient ID']:r for r in meta.to_dict('records')}
    events=json.loads((EVAL/'decoder_records.json').read_text())
    candidates={}
    for r in events:
        if r['role']!='pilot_validation' or r['budget_seconds']!=20:continue
        name=r['record'].split('/')[-1];pid,site=name.split('_',1)
        if site not in ['AV','PV','TV','MV']:continue
        entry=ids[pid]
        if entry['role']!='pilot_validation':raise AssertionError('Role mismatch')
        item=dict(r,group_id=entry['group_id'],patient_id=pid,site=site)
        candidates.setdefault(entry['group_id'],[]).append(item)
    selected=[]
    for group in sorted(candidates,key=lambda g:hashlib.sha256(('phase10-v1:'+g).encode()).hexdigest())[:24]:
        rows=sorted(candidates[group],key=lambda r:(['AV','PV','TV','MV'].index(r['site']),r['record']))
        selected.append(rows[0])
    if len(selected)!=24:raise AssertionError('Unexpected sample size')
    manifest=dict(status='selection_sealed_before_array_reads',groups=24,
                  labels_read=False,annotations_read=False,final_arrays_read=0,
                  sources={str(p.relative_to(ROOT)):digest(p) for p in
                           [SPLIT,EVAL/'decoder_records.json',SOURCE,CFG,PROTOCOL,Path(__file__)]},
                  records=selected)
    write(OUT/'selection_manifest.json',manifest)
    selection_sha=digest(OUT/'selection_manifest.json')
    # Fresh process, isolated module; no neural model, training labels or audio.
    capture=types.ModuleType('viterbi_hmm');captured=[]
    def hsmm_viterbi(p,d,D,a):
        captured.append((p.copy(),d.copy(),a.copy()))
        return np.zeros(len(p),dtype=np.int32)
    capture.hsmm_viterbi=hsmm_viterbi
    sys.modules['viterbi_hmm']=capture
    spec=importlib.util.spec_from_file_location('phase10_capture_segmenter',SOURCE)
    module=importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode=True
    spec.loader.exec_module(module)
    cfg=json.loads(CFG.read_text())
    outputs=[]
    for number,r in enumerate(selected):
        original=EVAL/'posteriors'/r['array']
        if digest(original)!=r['array_file_sha256']:raise AssertionError('Saved posterior changed')
        with np.load(original,allow_pickle=False) as z:
            p=z['posterior'].copy();old_h=z['healthy_states'].copy();old_m=z['murmur_states'].copy()
        if hashlib.sha256(p.tobytes()).hexdigest()!=r['posterior_raw_sha256']:raise AssertionError('Array digest differs')
        captured.clear();module.double_duration_viterbi(p,50,**cfg['hsmm'])
        if len(captured)!=4:raise AssertionError('Topology count changed')
        arrays=dict(posterior=p,legacy_healthy=old_h,legacy_murmur=old_m)
        for k,(q,d,a) in zip(NAMES,captured):
            arrays[k+'_posterior']=q;arrays[k+'_duration']=d;arrays[k+'_transition']=a
        path=OUT/f'{number:02d}.npz';np.savez_compressed(path,**arrays)
        outputs.append(dict(number=number,record=r['record'],group_id=r['group_id'],
                            array=path.name,sha256=digest(path),legacy=r['corrected']))
    for relative,expected in manifest['sources'].items():
        if digest(ROOT/relative)!=expected:raise AssertionError('Source changed during materialization')
    write(OUT/'materialization.json',dict(status='complete',selection_sha256=selection_sha,
          records=outputs,role='pilot_validation',budget_seconds=20,models=NAMES))
    print(json.dumps(dict(status='complete',records=len(outputs),selection_sha256=selection_sha)))

if __name__=='__main__':main()
