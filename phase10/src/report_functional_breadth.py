"""Plot all sixteen fixed breadth conditions; no model/condition selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'phase10/results/functional_breadth_v1'
OUT=ROOT/'phase10/results/functional_breadth_figures_v2'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists():raise FileExistsError(OUT)
    assert json.loads((SOURCE/'saved_result_audit.json').read_text())['status']=='passed'
    data=json.loads((SOURCE/'summary.json').read_text());rows=data['conditions'];OUT.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,(a,b)=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.15,1]});y=np.arange(16)
    labels=[]
    for r in rows:
        model=r['model'].replace('teacher_float','Float teacher').replace('teacher_int8','INT8 teacher').replace('hard_202611','Hard / seed ').replace('kd_202611','KD / seed ')
        labels.append(model+' / '+str(r['budget'])+(' s' if r['budget']==20 else ''))
    for key,label,color,marker in [('frame','Frame projection','#b8bdc5','s'),('segment','Segment projection','#627e96','o'),('fixed','Fixed ±1 queries','#c39a32','x'),('intersect','Functional + segment','#248979','D')]:
        x=[r['resolved'][key] for r in rows]
        a.scatter(x,y,label=label,c=color,marker=marker,s=38,zorder=3)
    for i,r in enumerate(rows):
        a.plot([r['resolved']['segment'],r['resolved']['intersect']],[i,i],c='#d2d9dc',lw=1,zorder=1)
        a.text(r['resolved']['intersect']+.9,i,str(r['resolved']['intersect']),va='center',fontsize=9,color='#1b6c60')
    a.set_yticks(y);a.set_yticklabels(labels);a.invert_yaxis();a.set_xlim(-1,50)
    a.set_xlabel('Resolved native site-score signs / 48');a.set_xticks([0,12,24,36,48]);a.grid(axis='x',alpha=.18)
    a.set_title('A  Sign resolution across every condition',loc='left',fontweight='bold',pad=15)
    a.legend(loc='lower left',bbox_to_anchor=(-.02,-.14),ncol=2,frameon=False,fontsize=9)
    reduction=np.array([r['median_paired_width_reduction_vs_segment']*100 for r in rows])
    b.barh(y,reduction,color='#248979',height=.62);b.set_yticks(y);b.set_yticklabels([]);b.invert_yaxis()
    for i,v in enumerate(reduction):b.text(v+1,i,f'{v:.1f}%',va='center',fontsize=9)
    b.set_xlim(0,100);b.set_xlabel('Median paired width reduction vs segment (%)');b.grid(axis='x',alpha=.18);b.set_axisbelow(True)
    b.axvline(50,color='#9ba3ab',lw=1,ls='--');b.set_title('B  Tighter functional bounds',loc='left',fontweight='bold',pad=15)
    fig.suptitle('Functional bounds retain their advantage in 48 additional people',x=.04,ha='left',fontsize=17,fontweight='bold',y=.98)
    fig.text(.04,.935,'Eight frozen predictors × two recording budgets · η = 0.005 · All 16 predeclared condition gates passed',fontsize=11,color='#44515e')
    fig.text(.04,.025,'Same 48 people repeated across conditions; 768 evaluations are not 768 independent participants.\nInternal development extension. Model-conditional path bounds; no disease accuracy or clinical validation.',fontsize=9,color='#44515e')
    fig.subplots_adjust(left=.205,right=.965,bottom=.18,top=.87,wspace=.20)
    for ext in ['png','svg']:fig.savefig(OUT/f'functional_breadth.{ext}',dpi=220,facecolor='white')
    (OUT/'caption.txt').write_text('On 48 participants disjoint from the initial diagnostic sample, functional bounds resolved 41-46 of 48 native site-score signs across sixteen fixed model/budget conditions, compared with 2-6 for segment projections. Median paired width reduction was 85.01-87.51% by condition. All conditions met the predeclared engineering gate. No clinical outcomes were analyzed.\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(status='generated_pending_visual_QA',sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),SOURCE/'summary.json',SOURCE/'saved_result_audit.json']},outputs={p.name:digest(p) for p in OUT.iterdir()}),indent=2)+'\n')
    print(OUT)

if __name__=='__main__':main()
