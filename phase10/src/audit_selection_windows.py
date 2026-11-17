"""Reconstruct each saved window bound from its reported dual multiplier."""
from pathlib import Path
import hashlib
import json
import numpy as np
from selection_aware_bounds import WindowOracle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'phase10/results/selection_aware_windows_v1'
INP=ROOT/'phase10/results/development_inputs_v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    run=json.loads((OUT/'run.json').read_text())
    for rel,sha in run['sources'].items():assert digest(ROOT/rel)==sha
    for rel,sha in run['outputs'].items():assert digest(OUT/rel)==sha
    details=json.loads((OUT/'oracle_details.json').read_text())
    material=json.loads((INP/'materialization.json').read_text())
    checked=0;max_error=0.;data={}
    for rec in material['records']:
        with np.load(INP/rec['array'],allow_pickle=False) as z:data[rec['number']]={k:z[k] for k in z.files}
    for row in details:
        z=data[row['number']];name=row['topology'];p=z[name+'_posterior'].astype(float)
        mask=np.zeros(len(p));mask[row['start_frame']:row['end_frame']]=1.
        oracle=WindowOracle(p,z[name+'_duration'],z[name+'_transition'],mask,.005)
        for kind in ['unconditional','conditioned']:
            for suffix,sign in [('minus',-1),('plus',1)]:
                key=kind+'_'+suffix
                if key not in row:continue
                saved=row[key];theta=saved['best_theta']
                elementary=float((np.max(sign*p,axis=1)*mask).sum()/len(p))
                if theta is None:reconstructed=elementary+1e-9
                else:
                    floor=row.get('confidence_floor',0.) if kind=='conditioned' else 0.
                    value,_,_=oracle.evaluate(theta,sign,floor)
                    reconstructed=min(elementary,value)+1e-9
                max_error=max(max_error,abs(saved['upper']-reconstructed))
                assert abs(saved['upper']-reconstructed)<1e-8
                checked+=1
    summary=dict(status='passed',support_bounds_reconstructed=checked,max_absolute_error=max_error,
        input_and_result_hashes_verified=True,window_rows=96,
        qualification='Independent saved-multiplier reconstruction; support DP itself checked by exhaustive synthetic references.',
        auditor_sha256=digest(Path(__file__)))
    target=ROOT/'phase10/results/selection_window_saved_bound_audit.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
