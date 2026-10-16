"""Standalone scientific figures for the frozen efficiency and selection results."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_surrogate_results import LABELS,sig

def main():
    analysis=json.loads((EXP/'results/final_selection_analysis.json').read_text());cost=json.loads((EXP/'results/surrogate_cost_comparison.json').read_text());selection_cost={sig(r):r for r in cost['selection_aggregates']}
    fig,axes=plt.subplots(1,3,figsize=(17,8.5));fig.subplots_adjust(bottom=.38,wspace=.28,top=.86)
    ax=axes[0];qq={(r['method'],r['steps']):r for r in analysis['direct_quality'] if r['family']=='fullpool'}
    short={'graph_oracle32_lr0.0003':'Graph Oracle32','oracle32_matched':'Oracle32','oracle16_matched':'Oracle16','cv16_matched':'CV16','subset16_lr0.0003':'Tuned Oracle16','cv16_lr0.0003':'Tuned CV16','zero_every4_lr0.001':'Zero / Adam','graph_cv8_lr0.001':'Graph CV8','graph_subset8_lr0.001':'Graph Oracle8','state256_cv16_until32_lr0.001':'State first32'}
    for i,c in enumerate(r for r in cost['direct_training'] if r['family']=='fullpool' and r['qualified_timing_adapters']==3):
        q=qq[c['method'],c['steps']];y=100*q['accuracy']['mean'];ci=np.array(q['accuracy']['ci95'])*100;marker='s' if c['kind']=='oracle' else '*' if c['kind']=='cv' else 'o'
        ax.errorbar(c['qualified_mean_preparation_plus_online_seconds'],y,yerr=[[y-ci[0]],[ci[1]-y]],fmt=marker,capsize=2,markersize=8,label=f"{short.get(c['method'],c['method'])} / {c['steps']}",alpha=.85)
    ax.set_title('Full-pool adaptation');ax.set_xlabel('Qualified preparation + updates (s)');ax.set_ylabel('Generated holdout accuracy (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=7.5,frameon=False)
    shortlist=[('oracle32_lr0.0003',128,'generated_accuracy','Oracle32 + generation'),('oracle32_matched',64,'generated_accuracy','Short Oracle + generation'),('cv16_matched',128,'generated_accuracy','CV16 + generation'),('cv16_matched',128,'mnli_behavior_anchored128_coverage:gold_probability','CV16 + behavior score'),('cv16_matched',128,'exact_teacher_gold_probability','CV16 + exact forced score'),('zero_every4_lr0.0003',256,'generated_accuracy','Zero / Adam screening')]
    for ax,family in zip(axes[1:],['random','genre']):
        rr=[r for r in analysis['groups'] if r['family']==family and r['target']=='all'];first=rr[0];random_accuracy=100*(first['selection_oracle_accuracy']['mean']-first['selection_delta_vs_random']['mean']);baseline_cost=float(np.mean([r['random_expected_training_seconds'] for r in cost['cheap_selection_baselines'] if r['family']==family]))
        reference=analysis['spec']['oracle_reference']
        reference_quality=[r for r in analysis['direct_quality'] if r['family']==family and all(r.get(k)==v for k,v in reference.items())]
        assert len(reference_quality)==1
        baseline=reference_quality[0]['accuracy'];assert abs(100*baseline['mean']-random_accuracy)<1e-8
        ci=np.array(baseline['ci95'])*100
        ax.errorbar(baseline_cost,random_accuracy,yerr=[[random_accuracy-ci[0]],[ci[1]-random_accuracy]],color='black',fmt='x',capsize=2,markersize=7,label='Random choice; no screening')
        ax.axhline(random_accuracy,color='gray',ls=':',lw=.8)
        if family=='random' and first.get('train_all_original_pool_accuracy'):
            baselines=[r['train_all_original_pool'] for r in cost['cheap_selection_baselines'] if r['family']=='random'];times=[r['qualified_preparation_plus_online_seconds'] for r in baselines if r['qualified_preparation_plus_online_seconds'] is not None]
            if len(times)==3:
                control=first['train_all_original_pool_accuracy'];y=100*control['mean'];ci=np.array(control['ci95'])*100
                ax.errorbar(np.mean(times),y,yerr=[[y-ci[0]],[ci[1]-y]],color='black',fmt='s',capsize=2,markersize=6,label='Train all 4,097 (more data)')

        for method,steps,selector,label in shortlist:
            found=[r for r in rr if r['method']==method and r['steps']==steps and r['selector']==selector];assert len(found)==1;r=found[0];c=selection_cost[sig(r)];y=100*r['selection_oracle_accuracy']['mean'];ci=np.array(r['selection_oracle_accuracy']['ci95'])*100
            ax.errorbar(c['mean_ready_component_sum_seconds'],y,yerr=[[y-ci[0]],[ci[1]-y]],fmt='o',capsize=2,markersize=5,label=label,alpha=.85)
        ax.set_title(f'{family.capitalize()} candidate selection');ax.set_xlabel('Selection + final Oracle component sum (s)');ax.set_ylabel('Chosen Oracle holdout accuracy (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=7.5,frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_xlim(left=0)
    fig.suptitle('Frozen recipes and new warmup starts · Qwen2.5-1.5B / MNLI',fontsize=15)
    fig.text(.04,.025,'95% intervals: paired start/source-group bootstrap; only three starts. Full-pool times use three exact-weight replays per start.\nSelection x-axes are component estimates, not measured end-to-end latency; final chosen Oracle training is included.\nThe whole-pool control trains on 4,097 examples; each candidate uses 1,024. Offline fitting, target warmup and unrecorded transitions are additional.\nEvaluation uses MNLI training-population group holdouts, not official validation.',fontsize=9,color='#4c5662')
    out=EXP/'results/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'frozen_surrogate_tradeoffs.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':main()
