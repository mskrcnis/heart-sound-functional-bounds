"""Rebuild manuscript tables and vector figures from sealed phase10 results.

No training, waveform access, label access, model selection, or result mutation.
Run from this checkout with .venv-foundation/bin/python.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PUB = Path(__file__).resolve().parents[1]
ROOT = PUB.parents[1]
RESULTS = ROOT / 'phase10/results'
used = {}
BUNDLE = '--from-bundle' in sys.argv

def read(rel):
    if BUNDLE:
        q=Path(rel)
        name='breadth_record_bounds_anonymized.json' if rel=='functional_breadth_v1/record_bounds.json' else q.parent.name+'__'+q.name
        p=PUB/'evidence'/name
        used['bundle/evidence/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
    else:
        p = RESULTS / rel
        used[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

def write_table(name, rows):
    (PUB / 'tables' / name).write_text('\n'.join(' & '.join(map(str,r)) + r' \\' for r in rows)+'\n')

def csvout(name, rows):
    with (PUB/'evidence'/name).open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

initial = read('functional_bounds_v1/summary.json')
segment = read('strong_baseline_audit_v1/audit.json')
cheap = read('cheap_queries_and_winners_v1/summary.json')
breadth = read('functional_breadth_v1/summary.json')
records = read('functional_breadth_v1/record_bounds.json')
tightness = read('functional_breadth_v1/tightness_diagnostic.json')
audit = read('functional_breadth_v1/saved_result_audit.json')
fresh = read('query_gap_training_v1/summary.json')
warm = read('query_gap_warmstart_v1/summary.json')
fa = read('query_gap_training_v1/independent_audit.json')
wa = read('query_gap_warmstart_v1/independent_audit.json')
windows = read('selection_aware_windows_v1/summary.json')
assert audit['status']==fa['status']==wa['status']=='passed'
assert len(records)==768 and len({r['group_id'] for r in records})==48
assert not fresh['engineering_gate_passed'] and not warm['engineering_gate_passed']
unknown = sum(r['intersect']['sign']=='unresolved' and not r['feasible_both_signs'] for r in records)
assert unknown == tightness['total_remaining_undetermined'] == 19
assert sum(r['intersect']['sign']!='unresolved' for r in records)==701
assert sum(r['feasible_both_signs'] for r in records)==48
for cond in breadth['conditions']:
    sub=[r for r in records if (r['model'],r['budget'])==(cond['model'],cond['budget'])]
    assert len(sub)==48
    for key in cond['resolved']:
        assert sum(r[key]['sign']!='unresolved' for r in sub)==cond['resolved'][key]
    widths=lambda key:np.array([r[key]['upper']-r[key]['lower'] for r in sub])
    assert np.isclose(np.median(1-widths('intersect')/widths('segment')),cond['median_paired_width_reduction_vs_segment'],atol=1e-12)
    assert sum(r['stable_sign_multiple_winners'] for r in sub)==cond['stable_sign_multiple_winners']

def model_label(s):
    return s.replace('teacher_float','Float teacher').replace('teacher_int8','INT8 teacher').replace('hard_202611','Hard / ').replace('kd_202611','KD / ')

rows=[]
for r in initial['results']:
    s=next(x for x in segment['results'] if x['eta']==r['eta'])
    rows.append([f"{r['eta']:g}",r['projected_certified'],s['segment_certified'],r['certified'],r['feasible_both_signs'],f"{r['median_outer_width']:.4f}", '--' if s['median_width_reduction_vs_segment'] is None else f"{100*s['median_width_reduction_vs_segment']:.2f}"])
write_table('initial_rows.tex',rows)
rows=[];bflat=[]
for r in breadth['conditions']:
    rows.append([model_label(r['model']),str(r['budget'])+(' s' if r['budget']==20 else ''),r['resolved']['segment'],r['resolved']['intersect'],r['resolved']['fixed'],f"{100*r['median_paired_width_reduction_vs_segment']:.2f}",r['stable_sign_multiple_winners']])
    bflat.append(dict(model=r['model'],budget=r['budget'],people=48,**{k+'_resolved':v for k,v in r['resolved'].items()},width_reduction=r['median_paired_width_reduction_vs_segment'],stable_sign_multiple_winners=r['stable_sign_multiple_winners'],witnessed_both_signs=r['witnessed_both_signs']))
write_table('breadth_rows.tex',rows);csvout('breadth_conditions.csv',bflat)
arm_labels={'kl':'Emission KL','score':'+ Score','confidence':'+ Confidence','query':'+ Support gaps'}
rows=[];allrows=[]
for label,data in [('Fresh',fresh),('Warm',warm)]:
    for arm,m in data['means'].items():
        rows.append([label,arm_labels[arm],f"{m['primary_bound_mae']:.6f}",f"{m['score_mae']:.6f}",f"{m.get('response_mae',0):.6f}" if 'response_mae' in m else '--',f"{m['emission_kl']:.6f}"])
    for r in data['all_arms']:
        m=r['metrics']
        allrows.append(dict(trial=label,seed=r['seed'],arm=r['arm'],people=m['n'],**{k:v for k,v in m.items() if k not in ['sign_agreement','n']},**{'agreement_'+k:v for k,v in m['sign_agreement'].items()}))
write_table('distillation_rows.tex',rows)
# CSV uses the same columns in both trials; response was not a trial-1 endpoint.
keys=list(dict.fromkeys(k for r in allrows for k in r))
csvout('distillation_all_21_models.csv',[{k:r.get(k,'') for k in keys} for r in allrows])
write_table('seed_rows.tex',[[r['trial'],str(r['seed'])[-2:],arm_labels[r['arm']],f"{r['primary_bound_mae']:.6f}",f"{r['global_bound_mae']:.6f}",f"{r['window_bound_mae']:.6f}",f"{r['emission_kl']:.6f}"] for r in allrows])
write_table('sign_rows.tex',[[r['trial'],str(r['seed'])[-2:],arm_labels[r['arm']]]+[f"{100*r['agreement_'+eta]:.1f}" for eta in ['0.001','0.005','0.01','0.05']] for r in allrows])
write_table('query_rows.tex',[[f"{r['eta']:g}",{'one':'1','three':'3','six':'6'}[r['query_set']],r['dp_calls'],r['sign_certified'],f"{r['median_width']:.5f}",f"{r['median_width_ratio_vs_full']:.3f}",f"{1000*r['median_seconds']:.2f}"] for r in cheap['query_results']])
write_table('tightness_rows.tex',[[model_label(r['model']),str(r['budget'])+(' s' if r['budget']==20 else ''),r['resolved'],r['proved_both_signs'],r['remaining_undetermined'],f"{r['median_outer_inner_gap']:.6f}",f"{r['max_outer_inner_gap']:.6f}"] for r in tightness['conditions']])
write_table('winner_rows.tex',[[f"{r['eta']:g}",r['outer_single_winner'],r['witnessed_multiple_winners'],r['stable_score_but_witnessed_multiple_winners']] for r in cheap['winner_results']])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
colors={'segment':'#6c7b8b','functional':'#147d75','fixed':'#c28a24','query':'#b34350'}
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(PUB/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(7.1,3.5));ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#b4c1ca',linewidth=.8))
    ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontweight='bold',fontsize=9)
    ax.text(x+w/2,y+h-.66,body,ha='center',va='top',fontsize=8.5,linespacing=1.35)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#657985',lw=1.2))
box(.1,2.65,2.5,1.55,'Frozen emissions','Fixed recording\nFour topology graphs','#edf2f6')
box(3.2,2.65,3.2,1.55,'Near-optimal paths',r'$L(q)\geq L^*-T\eta$'+'\nGlobal deficit budget','#e9f4f0')
box(7,2.65,2.85,1.55,'Functional oracle',r'$\max_q\{L(q)+\lambda R(q)\}$'+'\nBounds and witnesses','#e9f4f0')
box(2.2,.2,3.05,1.55,'Score sign','Positive / nonpositive\nor unresolved','#edf2f6')
box(6.2,.2,3.4,1.55,'Topology winner','Attainable alternatives\nExclusions and witnesses','#fbf0e5')
arrow((2.7,3.43),(3.1,3.43));arrow((6.5,3.43),(6.9,3.43));arrow((8.4,2.55),(7.9,1.85));arrow((7.3,2.55),(4.2,1.85))
ax.text(.1,4.75,'Model-conditional analysis of decision and explanation',fontsize=11,fontweight='bold')
save(fig,'method_overview')

fig,(a,b)=plt.subplots(1,2,figsize=(7.1,3.1))
etas=[r['eta'] for r in initial['results']][1:];xx=np.arange(len(etas))
for vals,label,col,marker in [([r['projected_certified'] for r in initial['results']][1:],'Frame','#a9b2bb','s'),([r['segment_certified'] for r in segment['results']][1:],'Segment',colors['segment'],'o'),([r['certified'] for r in initial['results']][1:],'Functional',colors['functional'],'D')]:
    a.plot(xx,vals,marker=marker,label=label,color=col,lw=1.5)
a.set_xticks(xx,etas);a.set_ylim(0,25);a.set_yticks([0,6,12,18,24]);a.set_xlabel(r'Normalized deficit $\eta$');a.set_ylabel('Resolved signs / 24');a.set_title('(a) Decision stability',loc='left');a.legend(frameon=False,fontsize=8)
vals=[r['stable_score_but_witnessed_multiple_winners'] for r in cheap['winner_results']]
b.bar(xx,vals,color=colors['functional'],width=.55);b.set_xticks(xx,etas);b.set_ylim(0,25);b.set_yticks([0,6,12,18,24]);b.set_xlabel(r'Normalized deficit $\eta$');b.set_ylabel('Stable sign + multiple winners / 24');b.set_title('(b) Explanation ambiguity',loc='left')
for x,v in zip(xx,vals):b.text(x,v+.5,str(v),ha='center')
for ax in [a,b]:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.8);save(fig,'initial_sensitivity')

fig,(a,b)=plt.subplots(1,2,figsize=(7.2,5.7),gridspec_kw={'width_ratios':[1.12,1]});yy=np.arange(16)
labels=[model_label(r['model'])+' / '+str(r['budget'])+(' s' if r['budget']==20 else '') for r in breadth['conditions']]
for key,label,col,marker,offset in [('frame','Frame','#adb6be','s',-.12),('segment','Segment',colors['segment'],'o',0),('fixed',r'Fixed $\pm1$',colors['fixed'],'x',-.10),('intersect','Functional',colors['functional'],'D',.10)]:
    a.scatter([r['resolved'][key] for r in breadth['conditions']],yy+offset,label=label,c=col,marker=marker,s=20,zorder=3)
for i,r in enumerate(breadth['conditions']):a.plot([r['resolved']['segment'],r['resolved']['intersect']],[i,i],c='#d8dfe2',lw=1,zorder=1)
a.set_yticks(yy,labels,fontsize=8);a.invert_yaxis();a.set_xlim(-1,49);a.set_xticks([0,12,24,36,48]);a.set_xlabel('Resolved signs / 48');a.set_title('(a) Sign resolution',loc='left');a.grid(axis='x',alpha=.2)
a.legend(loc='upper left',bbox_to_anchor=(-.65,-.11),ncol=4,frameon=False,fontsize=8)
red=np.array([r['median_paired_width_reduction_vs_segment']*100 for r in breadth['conditions']]);b.barh(yy,red,color=colors['functional'],height=.55)
b.set_yticks(yy,[]);b.invert_yaxis();b.set_xlim(0,100);b.set_xticks([0,25,50,75,100]);b.set_xlabel('Paired width reduction (%)');b.set_title('(b) Tightening vs segment',loc='left')
for i,v in enumerate(red):b.text(v+1,i,f'{v:.1f}',va='center',fontsize=7)
b.grid(axis='x',alpha=.18);b.set_axisbelow(True)
fig.subplots_adjust(left=.23,right=.98,bottom=.15,top=.94,wspace=.23);save(fig,'breadth')

fig,axes=plt.subplots(2,2,figsize=(7.2,5.0));ac={'kl':'#596f83','score':'#147d75','confidence':'#b58a2c','query':'#b34350'}
short={'kl':'KL','score':'+ Score','confidence':'+ Conf.','query':'+ Gaps'}
for ax,data,arms,key,title,scale,ylabel in [(axes[0,0],fresh,['kl','score','query'],'primary_bound_mae','(a) Fresh start',1,'Primary bound MAE'),(axes[0,1],warm,['kl','score','confidence','query'],'primary_bound_mae','(b) Common warm start',1,'Primary bound MAE'),(axes[1,0],warm,['kl','score','confidence','query'],'response_mae','(c) Warm: response fidelity',1000,r'Response MAE ($10^{-3}$)'),(axes[1,1],warm,['kl','score','confidence','query'],'emission_kl','(d) Warm: emission fidelity',1,'Temperature-scaled KL')]:
    for seed in sorted(data['per_seed']):
        v=[next(r['metrics'][key]*scale for r in data['all_arms'] if r['seed']==int(seed) and r['arm']==arm) for arm in arms];ax.plot(range(len(arms)),v,c='#cbd1d6',lw=.8,zorder=1)
    for i,arm in enumerate(arms):
        v=[r['metrics'][key]*scale for r in data['all_arms'] if r['arm']==arm];ax.scatter([i]*len(v),v,color=ac[arm],s=24,zorder=3);ax.plot([i-.13,i+.13],[np.mean(v)]*2,color=ac[arm],lw=2.3)
    ax.set_xticks(range(len(arms)),[short[k] for k in arms],fontsize=8);ax.set_ylabel(ylabel,fontsize=8);ax.set_title(title,loc='left',fontsize=9);ax.grid(axis='y',alpha=.18)
fig.tight_layout(w_pad=1.5,h_pad=2);save(fig,'distillation')

# Retain compact aggregate receipts. Raw identities, waveforms and weights are not copied.
if not BUNDLE:
    for rel in used:
        if rel.endswith('record_bounds.json'):continue
        p=ROOT/rel;shutil.copy2(p,PUB/'evidence'/(p.parent.name+'__'+p.name))
    ids={k:str(i+1) for i,k in enumerate(sorted({r['group_id'] for r in records}))}
    sanitized=[]
    for r in records:
        item={k:v for k,v in r.items() if k not in ['record','group_id','witnesses','witnesses_sha256']}
        item['group_id']=ids[r['group_id']]
        sanitized.append(item)
    (PUB/'evidence'/'breadth_record_bounds_anonymized.json').write_text(json.dumps(sanitized,indent=2)+'\n')
(PUB/'evidence'/'CORRECTION.txt').write_text('Publication audit, 2026-11-09\nThe earlier narrative NOVELTY_DECISION_v2.txt said 20 undetermined breadth cases. Direct recomputation from all 768 record_bounds.json entries gives 701 resolved + 48 witnessed both signs + 19 undetermined = 768. The saved tightness_diagnostic.json also gives 19. The manuscript uses 19. No saved scientific result was changed. These are repeated record-conditions, not independent people.\n')
(PUB/'evidence'/('bundle_rebuild_hashes.json' if BUNDLE else 'source_hashes.json')).write_text(json.dumps(used,indent=2)+'\n')
(PUB/'evidence'/'extraction_audit.json').write_text(json.dumps(dict(status='passed',people=48,record_conditions=768,resolved=701,witnessed_both_signs=48,undetermined=unknown,all_conditions_recomputed=True,all_21_fits_retained=True,source_files=len(used)),indent=2)+'\n')
print('Prepared tables, four vector figures, CSV evidence and correction; no experiment changes.')
