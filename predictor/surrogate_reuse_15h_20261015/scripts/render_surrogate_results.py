"""Render frozen numerical evidence; keep scientific interpretation in README."""
from common import *

LABELS={'graph_oracle32_lr0.0003':'Oracle32, graph, LR 0.0003','oracle32_matched':'Oracle32, LR 0.001','oracle32_lr0.0003':'Oracle32, LR 0.0003','oracle16_matched':'Oracle16, LR 0.001','subset16_lr0.0003':'Oracle16, LR 0.0003','cv16_matched':'Original CV16, LR 0.001','cv16_lr0.0003':'Original CV16, LR 0.0003','zero_every4_lr0.001':'Zero / Adam, true every 4, LR 0.001','zero_every4_lr0.0003':'Zero / Adam, true every 4, LR 0.0003','graph_cv8_lr0.001':'Original CV8, graph','graph_subset8_lr0.001':'Oracle8, graph','state256_cv16_until32_lr0.001':'State256 CV16 first 32, then Oracle16'}
WORKFLOW_LABELS={'oracle_generated':'Oracle32 + generation','oracle_forced':'Oracle32 + exact ABC score','cv16_generated':'CV16 + generation','cv16_forced':'CV16 + exact ABC score','cv16_behavior':'CV16 + learned behavior score','short_oracle_forced':'Short Oracle32 + exact ABC score'}
def interval(v):return f"{100*v['mean']:.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]"
def sig(r):return tuple(r[k] for k in ['family','method','kind','steps','lr','target','selector'])
def main():
    analysis=json.loads((EXP/'results/final_selection_analysis.json').read_text());cost=json.loads((EXP/'results/surrogate_cost_comparison.json').read_text());costmap={(r['family'],r['method'],r['steps']):r for r in cost['direct_training']};selectioncost={sig(r):r for r in cost['selection_aggregates']}
    lines=['# Frozen results','', 'Qwen2.5-1.5B-Instruct; MNLI training-population source-group holdouts. Three new warmup starts 302/303/304. All horizons count updates after 32 warmup updates. Evaluation inputs are complete; 256 is the maximum number of newly generated output tokens.','', 'Recipes were fixed before selection-dev outcomes. Candidate choices were then fixed before selection-holdout outcomes. Intervals resample starts and source groups and are descriptive with only three starts. A mean within 1 percentage point of Oracle is not a statistical noninferiority result.','']
    warmups=list(cost.get('shared_target_warmups',{}).values())
    if len(warmups)==3:
        lines += [f"The common 32-update target warmup averaged **{np.mean([r['online_seconds'] for r in warmups]):.2f}s** of updates and **{np.mean([r['recorded_wall_seconds'] for r in warmups]):.2f}s** under its recorded setup/update/save timer. These are one historical run per start, not three timing repetitions. Warmup is shared and excluded from the incremental costs below.",'']
    for family in ['fullpool','random','genre']:
        title={'fullpool':'Full 4,097-example adaptation pool','random':'Five 1,024-example random training subsets','genre':'Five 1,024-example source-disjoint genre training subsets'}[family]
        lines += [f'## {title}','', '| Method | Updates | Holdout accuracy %, 95% CI | Gap to population Oracle, pp, 95% CI | Gap to warmup, pp |','|---|---:|---:|---:|---:|']
        qq=[r for r in analysis['direct_quality'] if r['family']==family]
        for r in qq:
            lines.append(f"| {LABELS.get(r['method'],r['method'])} | {r['steps']} | {interval(r['accuracy'])} | {interval(r['delta_vs_oracle_same_candidate'])} | {100*r['delta_vs_warmup']['mean']:+.2f} |")
        warm=qq[0]['accuracy']['mean']-qq[0]['delta_vs_warmup']['mean'];lines += ['',f'Warmup-only mean generated accuracy: **{100*warm:.2f}%**. The fullpool reference is Oracle32/256/LR 0.001; candidate references are Oracle32/128/LR 0.0003. Candidate quality averages the five fixed subsets within each start; they are not independent seed replicates.','']
        lines += ['| Method | Updates | Qualified prep + updates, s | Qualified fresh process, s | Historical prep + updates, s | Recorded offline components, s | Profiled matrix PFLOPs |','|---|---:|---:|---:|---:|---:|---:|']
        for r in qq:
            c=costmap[(family,r['method'],r['steps'])]
            def num(k):return '—' if c[k] is None else f'{c[k]:.2f}'
            flop=c['profiled_matrix_flops'];flop='—' if not flop else f"{np.mean([x['flops'] for x in flop])/1e15:.4f}"
            lines.append(f"| {LABELS.get(r['method'],r['method'])} | {r['steps']} | {num('qualified_mean_preparation_plus_online_seconds')} | {num('qualified_mean_fresh_script_seconds')} | {c['historical_mean_preparation_plus_online_seconds']:.2f} | {c['offline']['recorded_component_sum_seconds']:.2f} | {flop} |")
        lines += ['', 'Qualified times average per-start medians of three fresh-process, exact-weight replays. Fullpool timing covers all three starts; candidate timing is representative **candidate0 only**, whereas historical costs cover all five subsets. Dashes indicate no qualified measurement. Fresh-process times include loading/imports/final save; both timing columns exclude the shared target warmup, offline construction and evaluation. Offline totals include recorded source warmups and dependencies, with historical timer boundaries. FLOPs are separately profiled on start 302 and count matrix operations, not all arithmetic.','']
    lines += ['## Additional paired controls','', '| Population | Actual recipe | Reference recipe | Accuracy difference, pp, 95% CI |','|---|---|---|---:|']
    for r in analysis.get('direct_paired_comparisons',[]):
        a,b=r['actual'],r['reference'];lines.append(f"| {r['family']} | {LABELS.get(a['method'],a['method'])} / {a['steps']} | {LABELS.get(b['method'],b['method'])} / {b['steps']} | {interval(r['delta'])} |")
    lines += ['', 'These paired comparisons separate predictor contribution, shorter/cheaper true-gradient alternatives, and the additional graph Oracle that had higher development accuracy. The primary population-specific Oracle references remain unchanged.','']
    lines += ['## Candidate-selection outcomes','', 'Each row selects one of five datasets using selection-dev, then reports the chosen **reference Oracle** agent on selection-holdout. This is distinct from deploying the predictor-updated agent directly. Cost estimates include the final reference-Oracle training run unless that exact adapter was already trained while screening.','']
    for family in ['random','genre']:
        lines += [f'### {family.capitalize()} candidates: aggregate target','', '| Simulator / readout | Updates | Chosen Oracle accuracy % | Gain vs random, pp, 95% CI | Gap vs Oracle-dev choice, pp, 95% CI | Regret to holdout ceiling, pp | Ready component sum, s |','|---|---:|---:|---:|---:|---:|---:|']
        for r in analysis['groups']:
            if r['family']!=family or r['target']!='all':continue
            if (r['method'],r['steps']) not in [('oracle32_lr0.0003',128),('oracle32_matched',64),('cv16_matched',128),('zero_every4_lr0.0003',256)]:continue
            c=selectioncost[sig(r)];label=r['selector'].replace('mnli_behavior_anchored128_coverage:','Anchored128 / ')
            lines.append(f"| {LABELS.get(r['method'],r['method'])} / {label} | {r['steps']} | {100*r['selection_oracle_accuracy']['mean']:.2f} | {interval(r['selection_delta_vs_random'])} | {interval(r['selection_delta_vs_oracle_dev_choice'])} | {100*r['regret_to_best_oracle_holdout_ceiling']['mean']:.2f} | {c['mean_ready_component_sum_seconds']:.2f} |")
        if family=='random':
            rows=[r for r in analysis['groups'] if r['family']=='random' and r['target']=='all'];control=rows[0].get('train_all_original_pool_accuracy')
            if control:
                base=[r['train_all_original_pool'] for r in cost['cheap_selection_baselines'] if r['family']=='random'];times=[r['qualified_preparation_plus_online_seconds'] for r in base if r['qualified_preparation_plus_online_seconds'] is not None]
                lines += ['',f"Training the original full 4,097-example pool once reaches **{100*control['mean']:.2f}%** without candidate screening. Qualified preparation/update time: **{np.mean(times):.2f}s**. This permits more examples than the 1,024-example candidate constraint; it is a practical alternative, not a same-data-size ablation." if len(times)==3 else f"Training the original full 4,097-example pool once reaches **{100*control['mean']:.2f}%** without screening; qualified timing is pending. This is not a same-data-size ablation."]
        lines += ['', 'Ready selection costs are **recorded component sums, not measured end-to-end pipeline latency**. Shared engine/model startup, process transitions and unrecorded I/O remain additional; fitted-predictor costs are separate in the cost ledger. Forced ABC scores and free-generation accuracy use different readouts; exact-teacher forced-score controls are included. The best-holdout candidate is a nondeployable diagnostic ceiling.','']
    lines += ['## Keeping the screened adapter directly','', 'This uses the already-frozen secondary `direct_deploy_accuracy` endpoint: choose on selection-dev and deploy that simulator-produced adapter, without a final reference-Oracle retraining. All rows below use generated-accuracy selection. The chosen dataset and its adapter are fixed before holdout readouts. This answers a different deployment question from the chosen-reference-Oracle table above.','', '| Candidates | Screening method | Updates | Deployed accuracy % | Gap vs Oracle-dev selected agent, pp, 95% CI | Ready component sum without final retraining, s | Offline components, s |','|---|---|---:|---:|---:|---:|---:|']
    for r in analysis['groups']:
        if r['target']!='all' or r['selector']!='generated_accuracy':continue
        if (r['method'],r['steps']) not in [('oracle32_lr0.0003',128),('oracle32_matched',64),('oracle16_matched',128),('cv16_matched',128),('zero_every4_lr0.0003',256)]:continue
        c=selectioncost[sig(r)];seconds=c['mean_ready_component_sum_seconds']-c['mean_final_reference_training_component_seconds']
        lines.append(f"| {r['family']} | {LABELS.get(r['method'],r['method'])} | {r['steps']} | {100*r['direct_deploy_accuracy']['mean']:.2f} | {interval(r['direct_deploy_delta_vs_oracle_dev_choice'])} | {seconds:.2f} | {c['recorded_offline_component_sum_seconds']:.2f} |")
    lines += ['', 'These costs subtract the separately recorded final-Oracle component from the existing component ledger; they are not separately measured complete-workflow latencies. The measured workflow benchmark below retains its registered final-reference-Oracle endpoint. Reporting these frozen secondary results does not select a new winning method from holdout outcomes.','']
    workflow=cost.get('measured_selection_pipeline')
    if workflow and workflow['complete']:
        lines += ['## Measured complete selection workflows','', 'One representative start 302 and the five random candidates; three independent timing repetitions per condition. Every candidate adaptation is rerun, every readout is freshly computed, and the final chosen Oracle is rerun when necessary. Final weights, readouts and choices must exactly reproduce the frozen references.','', '| Workflow | Median total seconds | Range, seconds | Repeats |','|---|---:|---:|---:|']
        for r in workflow['rows']:
            lines.append(f"| {WORKFLOW_LABELS.get(r['condition'],r['condition'])} | {r['pipeline_seconds_median']:.2f} | {r['pipeline_seconds_min']:.2f}–{r['pipeline_seconds_max']:.2f} | {r['qualified_repetitions']} |")
        lines += ['', 'These are actual fresh-process workflow times, including loading, process transitions, serialization and equality checks. Offline predictor construction and the common target warmup remain additional. The implementation launches separate adaptation/readout processes; it is not an optimized persistent model server. Three timing repetitions are not three scientific starts. Genre workflows are not timed end-to-end.','']
    fidelity_path=EXP/'results/final_behavior_fidelity.json'
    if fidelity_path.exists():
        fidelity=json.loads(fidelity_path.read_text());lines += ['## Predicting changed agent behavior','', '| Fullpool agent | Updates | Exact teacher % | Behavior surrogate % | Unchanged initial % | Agreement gain vs unchanged, pp, 95% CI |','|---|---:|---:|---:|---:|---:|']
        for r in fidelity['rows']:
            if r['family']!='fullpool' or (r['method'],r['steps']) not in [('oracle32_matched',256),('cv16_matched',256),('graph_cv8_lr0.001',512),('state256_cv16_until32_lr0.001',256)]:continue
            m=r['metrics'];lines.append(f"| {LABELS.get(r['method'],r['method'])} | {r['steps']} | {100*m['teacher_accuracy']['mean']:.2f} | {100*m['surrogate_accuracy']['mean']:.2f} | {100*m['initial_accuracy']['mean']:.2f} | {interval(m['agreement_delta_vs_initial'])} |")
        lines += ['', 'Forced-format ABC outcomes only. Positive agreement gain means better prediction of the current teacher than simply retaining the initial teacher. Full logit-error and class-change diagnostics are in [behavior fidelity](results/final_behavior_fidelity.json).','']
    lines += ['All method/horizon/readout/genre-target results, candidate spreads and per-start choices are in [selection analysis](results/final_selection_analysis.json), [cost ledger](results/surrogate_cost_comparison.json), and [qualified timing receipts](results/standalone_reuse_timing_summary.json). Cheap constant, random and matching-genre metadata selectors are included in those files.','']
    (EXP/'report/RESULTS.md').write_text('\n'.join(lines));log(dict(direct_rows=len(analysis['direct_quality']),selection_groups=len(analysis['groups'])))

if __name__=='__main__':main()
