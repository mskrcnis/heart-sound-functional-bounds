"""Build an aggregate-only exploratory research note and diagnostic figure."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'phase10'
def read(rel):return json.loads((P/rel).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=P/'results/figures_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    full=read('results/functional_bounds_v1/summary.json')
    strong=read('results/strong_baseline_audit_v1/audit.json')
    cheap=read('results/cheap_queries_and_winners_v1/summary.json')
    windows=read('results/selection_aware_windows_v1/summary.json')
    records=read('results/functional_bounds_v1/record_bounds.json')
    wrows=read('results/selection_aware_windows_v1/window_bounds.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    etas=np.array([.001,.005,.01,.05]);xx=np.arange(4)
    f=[r for r in full['results'] if r['eta']>0];s=[r for r in strong['results'] if r['eta']>0]
    ax=axes[0,0]
    for label,values,color,marker in [
        ('State projections',[r['projected_certified'] for r in f],'#9CA3AF','s'),
        ('Segment/transition projections',[r['segment_certified'] for r in s],'#D97706','^'),
        ('Decision-functional bounds',[r['certified'] for r in f],'#126B91','o')]:
        ax.plot(xx,values,marker=marker,label=label,color=color,lw=1.8)
    ax.set(xticks=xx,xticklabels=[str(v) for v in etas],ylim=(-.5,25),yticks=[0,6,12,18,24],
           xlabel='Normalized log-score tolerance',ylabel='Recordings with resolved score sign',
           title='A  Decision uncertainty: 24 fixed records')
    ax.legend(fontsize=8,frameon=False);ax.grid(axis='y',alpha=.2)
    ax=axes[0,1];r=sorted([x for x in records if x['eta']==.005],key=lambda x:x['base_score'])
    for i,v in enumerate(r):
        color='#B8493A' if v['sign']=='unresolved' else '#126B91'
        ax.plot([v['lower'],v['upper']],[i,i],color=color,lw=2)
        ax.scatter(v['base_score'],i,s=13,color=color,zorder=3)
    ax.axvline(0,color='black',lw=.8,ls='--')
    ax.set(xlabel='Native site confidence contrast',ylabel='Recordings ordered by point score',
           yticks=[],title='B  Bounds at tolerance 0.005')
    ax.text(.98,.04,'4 intervals cross zero;\nboth signs have feasible witnesses',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    ax=axes[1,0];winner=cheap['winner_results']
    values=[r['stable_score_but_witnessed_multiple_winners'] for r in winner]
    ax.bar(xx,values,color='#126B91',width=.55)
    for i,v in enumerate(values):ax.text(i,v+.3,str(v),ha='center')
    ax.set(xticks=xx,xticklabels=[str(v) for v in etas],ylim=(0,15),yticks=[0,3,6,9,12,15],
           xlabel='Normalized log-score tolerance',ylabel='Recordings (of 24)',
           title='C  Stable score sign, competing explanations')
    ax.grid(axis='y',alpha=.2)
    ax=axes[1,1]
    red=np.array([100*(1-(r['selection_upper']-r['selection_lower'])/
                     (r['pruned_upper']-r['pruned_lower'])) for r in wrows])
    ax.hist(red,bins=np.linspace(0,30,16),color='#7A5195',edgecolor='white')
    ax.axvline(np.median(red),color='#352345',ls='--',lw=1.4)
    ax.text(.96,.92,f'Median: {np.median(red):.2f}%\n96 fixed quarter-windows',
            transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax.set(xlabel='Extra width reduction beyond topology pruning (%)',ylabel='Windows',
           title='D  Limited gain from the added selection constraint')
    fig.suptitle('Exploratory decision and explanation uncertainty — no clinical outcome assessment',fontsize=12)
    for ext in ['png','svg']:fig.savefig(out/f'functional_uncertainty_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    caption=(
      'Fixed label-free development sample: 24 recordings from 24 linked people, one canonical '
      'site each, short pure-KD seed 20261107, up to 20 seconds/site. These are model-conditional '
      'outer bounds over a declared log-score path tolerance, not statistical confidence intervals '
      'or disease probabilities. Panel A compares bounds derived from local states, retained '
      'segment/transition edges and the full path-budget dual. Panel B uses the float64 legacy '
      'boundary DAG; red denotes unresolved native site-score sign, not clinical referral. '
      'Panel C requires constructive alternative winning topologies. Panel D reports all four '
      'fixed quarter-windows per record; extra conditioning gives limited median benefit. '
      'Selection/baseline follow-ups were adaptive; no untouched clinical confirmation is claimed.\n')
    (out/'caption.txt').write_text(caption)
    manifests={p.name:digest(p) for p in out.iterdir()}
    (out/'manifest.json').write_text(json.dumps(dict(status='awaiting_visual_QA',files=manifests),indent=2)+'\n')
    print(out/'functional_uncertainty_diagnostic.png')

if __name__=='__main__':main()
