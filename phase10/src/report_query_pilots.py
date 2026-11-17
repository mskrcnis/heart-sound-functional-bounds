"""Scientific diagnostic figure retaining both failed distillation experiments."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FIRST=ROOT/'phase10/results/query_gap_training_v1'
SECOND=ROOT/'phase10/results/query_gap_warmstart_v1'
OUT=ROOT/'phase10/results/query_gap_figures_v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())

def main():
    if OUT.exists():raise FileExistsError(OUT)
    for source in [FIRST,SECOND]:assert load(source/'independent_audit.json')['status']=='passed'
    a=load(FIRST/'summary.json');b=load(SECOND/'summary.json')
    assert not a['engineering_gate_passed'] and not b['engineering_gate_passed']
    OUT.mkdir();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                                    'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8));colors={'kl':'#52667a','score':'#34836d','confidence':'#b48b35','query':'#bd4f56'}
    labels={'kl':'Emission KL','score':'+ Score','confidence':'+ Confidence','query':'+ Support gap'}
    def panel(ax,data,arms,key,title,ylabel,scale=1):
        seeds=sorted(set(r['seed'] for r in data['all_arms']))
        for seed in seeds:
            values=[next(r['metrics'][key]*scale for r in data['all_arms'] if r['seed']==seed and r['arm']==arm) for arm in arms]
            ax.plot(range(len(arms)),values,color='#c7cdd3',lw=1,zorder=1)
        for i,arm in enumerate(arms):
            values=[r['metrics'][key]*scale for r in data['all_arms'] if r['arm']==arm]
            ax.scatter(np.full(len(values),i),values,color=colors[arm],s=37,zorder=3)
            ax.plot([i-.13,i+.13],[np.mean(values)]*2,color=colors[arm],lw=3,zorder=4)
        ax.set_xticks(range(len(arms)));ax.set_xticklabels([labels[k] for k in arms]);ax.set_ylabel(ylabel)
        ax.set_title(title,loc='left',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    panel(axes[0,0],a,['kl','score','query'],'primary_bound_mae','A  Fresh start: bound fidelity','Primary bound MAE')
    panel(axes[0,1],b,['kl','score','confidence','query'],'primary_bound_mae','B  Common warm start: bound fidelity','Primary bound MAE')
    panel(axes[1,0],b,['kl','score','confidence','query'],'response_mae','C  Non-nominal uncertainty response','Response MAE (×10⁻³)',1000)
    panel(axes[1,1],b,['kl','score','confidence','query'],'emission_kl','D  Frame-distribution fidelity','Temperature-scaled emission KL')
    fig.suptitle('Support-gap distillation did not pass either development gate',x=.07,ha='left',fontsize=17,fontweight='bold',y=.985)
    fig.text(.07,.943,'80 fit / 16 monitor / 24 development people · 3 paired seeds · 4,877 parameters · Lower is better',fontsize=11,color='#44515e')
    fig.text(.07,.02,'Points: seed means over the same 24 people; short bars: three-seed mean. Lines connect matched seeds.\nA: 20 epochs from random initialization, auxiliary weight 10. B–D: adaptive 20-epoch continuation, weight 1. No clinical outcomes.',fontsize=9,color='#44515e')
    fig.subplots_adjust(top=.87,bottom=.13,hspace=.40,wspace=.24)
    for ext in ['png','svg']:fig.savefig(OUT/f'distillation_falsification.{ext}',dpi=220,facecolor='white')
    caption='Both frozen engineering gates failed. From random starts, support-gap matching worsened bound and emission fidelity. After a common KL warm start and reduced auxiliary weight, it improved primary error by 1.77% versus continued KL but remained 4.62% worse than score matching. Its 0.33% advantage over nominal-confidence matching did not extend to the uncertainty response, whose MAE was 2.13% worse. These are adaptive, small-sample teacher-fidelity findings, not clinical results.'
    (OUT/'caption.txt').write_text(caption+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(status='generated_pending_visual_QA',sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),FIRST/'summary.json',SECOND/'summary.json',FIRST/'independent_audit.json',SECOND/'independent_audit.json']},outputs={p.name:digest(p) for p in OUT.iterdir()}),indent=2)+'\n')
    print(OUT)

if __name__=='__main__':main()
