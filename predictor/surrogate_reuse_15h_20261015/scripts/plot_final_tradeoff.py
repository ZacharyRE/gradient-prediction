"""Frozen paired accuracy gaps versus verified standalone adaptation times."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def main():
    data=json.loads((EXP/'results/final_comparison.json').read_text())
    timings=json.loads((EXP/'results/frozen_timing_analysis.json').read_text())
    plan=json.loads((EXP/'configs/final_selection_plan.json').read_text())
    labels={(r['task'],r['cohort']):r.get('label',r['cohort'].replace('_',' ')) for r in plan['cohorts']}
    time_rows={(r['task'],r['cohort']):r for r in timings['rows']}
    fig,axes=plt.subplots(1,2,figsize=(13,7.5));fig.subplots_adjust(bottom=.35,top=.87,wspace=.23)
    excluded=[]
    for ax,task in zip(axes,['paws','mnli']):
        rows=[r for r in data['rows'] if r['task']==task and r['cohort']!='warmup']
        eligible=[r for r in rows if r['eligible_for_direct_frozen_quality_speed_claim']]
        excluded += [dict(task=task,cohort=r['cohort']) for r in rows if r not in eligible]
        assert eligible,'Require measured, weight-matched timing replays'
        lo=min([-1.]+[r['gap_ci95_low_pp'] for r in eligible])-.3
        hi=max([.2]+[r['gap_ci95_high_pp'] for r in eligible])+.3
        ax.axhspan(-1,hi,color='#eef6ef',zorder=0)
        ax.axhline(0,color='#98a0a8',lw=.8)
        ax.axhline(-1,color='#52765b',ls='--',lw=1)
        colors={'oracle_cached':'#7e8b96','oracle16':'#7e8b96','oracle_early':'#d97726','oracle8':'#8559a6','cv4':'#008477','cv16':'#8a623d','cv8':'#b85670','cv8_long':'#b85670','hold2':'#9a8218','anchor2':'#8559a6'}
        for index,r in enumerate(eligible):
            reference=r['cohort']==r['reference'];primary=r['primary_candidate']
            color='#222b35' if reference else '#2563eb' if primary else colors[r['cohort']]
            marker='s' if reference else '*' if primary else 'o'
            tt=time_rows[task,r['cohort']]['online_plus_preparation_seconds']
            x,y=tt['median'],r['gap_pp']
            ax.vlines(x,r['gap_ci95_low_pp'],r['gap_ci95_high_pp'],color=color,lw=1.3)
            ax.hlines(y,tt['minimum'],tt['maximum'],color=color,lw=1.3)
            label=f"{labels[task,r['cohort']]} · {r['update_steps']} updates"
            ax.scatter(x,y,color=color,marker=marker,s=105 if primary else 43,label=label,zorder=3)
        ax.set_ylim(lo,hi);ax.set_xlim(left=0)
        ax.set_title(task.upper(),fontsize=12)
        ax.set_xlabel('Online updates + required preparation (seconds)')
        ax.set_ylabel('Accuracy gap to selected Oracle32 (percentage points)')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%+.1f'))
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=7.8,frameon=False,columnspacing=1.)
    fig.suptitle('Qwen2.5-1.5B · frozen test quality and measured adaptation time',fontsize=14,y=.98)
    fig.text(.04,.04,'Stars: preselected primary candidates. Vertical intervals: paired group/seed bootstrap (95%). Horizontal ranges: three timing replays.\nDashed line: −1 pp practical margin; intervals crossing it leave matching quality uncertain.\nExisting fitted predictors; offline fitting, shared warmup and evaluation costs are additional. Test evidence is exploratory.',fontsize=8.5,color='#525a65')
    out=EXP/'results/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'final_tradeoff.{ext}',dpi=175)
    plt.close(fig)
    log(dict(plotted=sum(r['eligible_for_direct_frozen_quality_speed_claim'] for r in data['rows']),excluded_unverified_timings=excluded))


if __name__=='__main__':main()
