"""Standalone final primary-effect figures and exact CSV data; no model execution."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S=Path(__file__).resolve().parents[1]
planpath=S/'results/confirmation_plan.json'
plan=json.loads(planpath.read_text())
primary=plan['primary_family'];seeds=plan['seeds']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
figdir=S/'figures';figdir.mkdir(exist_ok=True)
for scoring,suffix in [('primary_numeric_repaired',''),('strict_numeric_repaired','_strict')]:
    source=S/f'results/confirmation_analysis{suffix}.json'
    result=json.loads(source.read_text());assert result['plan_sha256']==sha(planpath)
    math=result['datasets'][plan['primary_dataset']]['families'][primary]['vs_base']
    ood=result['ood_macro'][primary]
    panels=[('MATH test: 5,000 questions (historically reused)',math),
            ('Reserved mathematical OOD: equal task macro',ood)]
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.4),sharey=True)
    rows=[]
    for ax,(title,stats),dataset in zip(axes,panels,[plan['primary_dataset'],'ood_macro']):
        intervals=[stats['question_ci95'],stats['seed_question_ci95']]
        values=stats['per_seed_delta_pp'];assert len(values)==len(seeds)
        endpoints=[0,*values,*intervals[0],*intervals[1]]
        lo,hi=min(endpoints),max(endpoints);pad=max(.6,(hi-lo)*.2)
        ax.set_xlim(lo-pad,hi+pad)
        ax.axvline(0,color='#8d8d8d',lw=1,ls='--',zorder=0)
        ax.axhline(5.0,color='#dddddd',lw=.8,zorder=0)
        for y,(seed,value) in enumerate(zip(seeds,values)):
            color='#237a70' if value>0 else '#b7632f'
            ax.plot(value,y,'o',color=color,ms=7,zorder=3)
            ax.annotate(f'{value:+.2f}',(value,y),xytext=(7,0),textcoords='offset points',
                        va='center',fontsize=9,color=color)
            rows.append(dict(scoring=scoring,metric=dataset,kind='training_seed',seed=seed,
                             delta_pp=value,ci_low_pp='',ci_high_pp=''))
        for y,kind,ci in zip([6,7],['question_bootstrap','seed_question_bootstrap'],intervals):
            ax.hlines(y,ci[0],ci[1],color='#26364a',lw=2)
            ax.vlines(ci,[y-.08,y-.08],[y+.08,y+.08],color='#26364a',lw=1)
            ax.plot(stats['delta_pp'],y,'D',color='#26364a',ms=5)
            ax.annotate(f"{stats['delta_pp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
                        (stats['delta_pp'],y),xytext=(0,12),textcoords='offset points',
                        ha='center',fontsize=8,color='#26364a')
            rows.append(dict(scoring=scoring,metric=dataset,kind=kind,seed='',
                             delta_pp=stats['delta_pp'],ci_low_pp=ci[0],ci_high_pp=ci[1]))
        ax.set_title(title,fontsize=11,pad=12)
        ax.set_xlabel('Accuracy change versus the same baseline (percentage points)',fontsize=9)
        ax.set_yticks([0,1,2,3,4,6,7],
            [*[f'Seed {s}' for s in seeds],'Mean: question 95% CI','Mean: seed + question 95% CI'])
        ax.tick_params(axis='y',labelsize=9)
        ax.grid(axis='x',color='#eeeeee',lw=.7)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    axes[0].set_ylim(7.6,-.65)
    label=primary.replace('_',' ')
    fig.suptitle(f'Frozen primary recipe: {label}',fontsize=14,y=.98)
    fig.text(.5,.015,f'{scoring.replace("_"," ")} scoring. Five fixed training seeds; points are individual runs. '
             'Intervals do not imply success on every possible seed or task.',ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.05,1,.93],w_pad=2.5)
    stem=figdir/f'confirmation_primary{suffix}'
    for extension in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+extension),dpi=190,bbox_inches='tight')
    plt.close(fig)
    with stem.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stem.with_suffix('.sources.json').write_text(json.dumps(dict(
        plan=str(planpath),plan_sha256=sha(planpath),analysis=str(source),analysis_sha256=sha(source),
        script_sha256=sha(__file__),primary_family=primary,scoring=scoring,criteria=result['criteria']),indent=2)+'\n')
    print(stem)
