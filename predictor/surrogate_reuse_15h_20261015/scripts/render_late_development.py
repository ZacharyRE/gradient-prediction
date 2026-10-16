"""Render bounded post-freeze ablations without mixing them into final results."""
from common import *

def main():
    result=json.loads((EXP/'results/late_development_summary.json').read_text())
    catalog=json.loads((EXP/'results/surrogate_preparation_components.json').read_text())
    costs={r['predictor']:r for r in catalog['predictors']}
    rows={(r['method'],r['steps']):r for r in result['groups']}
    methods=[('512 inputs, width 256','Full CV16','diversity512_control_cv16_lr0.001'),
             ('512 inputs, width 256','CV16 first 32, then Oracle16','diversity512_control_cv16_until32_lr0.001'),
             ('2,048 inputs, width 256','Full CV16','diversity2048_cv16_lr0.001'),
             ('2,048 inputs, width 256','CV16 first 32, then Oracle16','diversity2048_cv16_until32_lr0.001'),
             ('2,048 inputs, width 512','Full CV16','capacity512_cv16_lr0.001'),
             ('2,048 inputs, width 512','CV16 first 32, then Oracle16','capacity512_cv16_until32_lr0.001'),
             ('2,048 inputs, width 512','Full CV8','capacity512_cv8_lr0.001')]
    table=['| Fitting inputs / width | Update policy | Accuracy at 256 | Seconds at 256 | Accuracy at 512 | Seconds at 512 |',
           '|---|---|---:|---:|---:|---:|']
    for label,policy,name in methods:
        rr=[rows[name,s] for s in [256,512]]
        assert all(sorted(r['starts'])==[301,311,321] for r in rr)
        values=[f"{r['mean_accuracy_pct']:.2f}% | {r['mean_preparation_updates_seconds']:.2f}" for r in rr]
        table.append(f'| {label} | {policy} | '+ ' | '.join(values)+' |')
    fits=['| Model | Epochs | Optimizer steps | Selected residual ratio | Label collection, s | Fit, s | Construction components, s |',
          '|---|---:|---:|---:|---:|---:|---:|']
    diagnostic=['| Model | Oracle 256 residual ratio | Oracle 512 residual ratio |', '|---|---:|---:|']
    for name in ['state256_coverage_fast','state256_diversity2048','state512_diversity2048']:
        model=result['fitting'][name];s=model['summary'];c=model['config'];d=model['selected_development_diagnostics']['states'];cost=costs['mnli_critic_'+name]
        fits.append(f"| {name} | {c['job']['epochs']} | {s['optimizer_steps']:,} | {s['best_centered_cv_ratio']:.4f} | {s['collection_cost_seconds']:.1f} | {s['seconds']:.1f} | {cost['recorded_component_sum_seconds']:.1f} |")
        diagnostic.append(f"| {name} | {d['w301_fast_step256']['centered_cv_ratio']:.4f} | {d['w301_fast_step512']['centered_cv_ratio']:.4f} |")
    pair={(r['early_cutoff'],r['steps']):r for r in result['diversity_paired']}
    assert result['diversity_fit_audit']['passed'] and all(r['complete'] for r in pair.values())
    gate=json.loads((EXP/'results/late_confirmation_gate_decision.json').read_text())
    gate_text=('One existing recipe passed the prospectively registered gate; its independent confirmation is reported separately.' if gate['triggered']
               else 'No new point passed both the 83.45% development-accuracy and 27.43s training-time thresholds. This gate was registered during late development, after some pilot readouts and before the complete three-start results. The conditional independent confirmation was not launched.' if not gate['passing_rows']
               else 'A point passed the quality/time thresholds, but the remaining confirmation budget was insufficient; no independent confirmation was launched.')
    text=f'''# Post-freeze development: more inputs and model capacity

These exploratory results do **not** replace the original frozen 435-entry comparison. They use the same 513 development examples, starts 301/311/321 with update seeds 401/411/421, the full 4,097-example training pool and LR 0.001. Time is historical prefix preparation plus updates, excluding warmup, offline fitting and evaluation; these are not three standalone timing repetitions.

## Downstream results

{chr(10).join(table)}

For comparison, the previously selected native Oracle32 reaches **83.95% at 256 updates / 27.43s** on these three development starts. Increasing fitting diversity changes the early-use 512-step mean by **{pair[True,512]['mean_difference_pp']:+.2f}pp** versus its 512-input control, but changes full-CV16 at 512 by only **{pair[False,512]['mean_difference_pp']:+.2f}pp**. At 256 the differences are **{pair[True,256]['mean_difference_pp']:+.2f}pp** for early use and **{pair[False,256]['mean_difference_pp']:+.2f}pp** for full CV16. Improving one long-horizon checkpoint does not establish a better time/quality frontier. The early-use recipe performs no predictor inference after step 32.

{gate_text}

## Matched fitting exposure and actual cost

All three fits use the same 12 development teacher states. The 512-input/40-epoch and 2,048-input/10-epoch fits each visit **245,760 example-state pairs**. The 512 rows form an exact nested subset of the 2,048 rows; all original training shards and predictor-dev shards are byte-identical in the expanded cache. Optimizer-step counts differ slightly because full sequences are packed into batches. The original inherited p123 predictor already used 2,050 inputs; this ablation concerns the new state-aware architecture.

{chr(10).join(fits)}

Construction components include the recorded label collection, complete fit, prerequisite teacher trajectories and source warmups. Shared dependencies are charged once per model; do not sum model rows. These are historical receipts, not clean cold-start measurements. Imports and unrecorded I/O remain additional. The fresh 512-input control at start 301 exactly reproduced all four original coverage checkpoints.

{chr(10).join(diagnostic)}

Residual ratio compares centered prediction error variance with true-gradient variance on 129 predictor-development examples. These states also participate in fitting/selection. Values near 1 at 512 indicate little variance reduction on this diagnostic, not a theorem about Adam or downstream quality. Better selected gradient metrics did not consistently improve actual rollouts.

## Bounded token-distillation readout

The three previously registered KD fits use the original 8-state/512-input cache. Their readout is a single-start development screen at start 301, not an independent confirmation. Pure token-critic updates use a restricted answer-token objective; CV16 retains the original teacher correction objective.

'''
    kd=[r for r in result['groups'] if r['method'].startswith('token_kd') and r['steps']==256]
    if kd:
        text+='| Recipe | Accuracy at 256 | Prep + updates, s |\n|---|---:|---:|\n'
        for r in kd:text+=f"| {r['method']} | {r['mean_accuracy_pct']:.2f}% | {r['mean_preparation_updates_seconds']:.2f} |\n"
        text+='\n| Critic | Selected epoch / 50 | Complete fit, s | Recorded construction components, s |\n|---|---:|---:|---:|\n'
        for name in ['token_kd128','token_kd128_nograd','token_kd256']:
            s=result['fitting'][name]['summary'];c=costs['mnli_critic_'+name]
            text+=f"| {name} | {s['best_epoch']} / 50 | {s['seconds']:.1f} | {c['recorded_component_sum_seconds']:.1f} |\n"
        text+='\nConstruction charges existing source states, cached labels and the full fit; shared dependencies are deduplicated within each model, so rows must not be added together.\n'
    else:text+='No completed downstream readout is available; fitting alone is not counted as downstream success.\n'
    token=[r for r in result['groups'] if r['method'].startswith('token_') and not r['method'].startswith('token_kd') and r['steps']==256]
    if len(token)==24:
        import csv
        warm=next(r for r in csv.DictReader((EXP/'results/development_summary.csv').open()) if r['tag']=='mnli_warmup_w301')
        short=rows['needed_oracle32_s32',32]
        text+='\n## Other token-critic configurations\n\n'
        text+='Four previously fitted token critics were screened with two learning rates and three update policies at start 301. Each row below is the **best observed development accuracy among eight recipes**, selected after seeing these outcomes, with no independent confirmation. All 24 recipes and their 32/128/256-step readouts remain in the numerical inventory.\n\n'
        text+='| Policy | Best observed recipe | Accuracy at 256 | Prep + updates, s | Accuracy range across eight recipes |\n|---|---|---:|---:|---:|\n'
        for label,key in [('CV4','_cv4_'),('CV16','_cv16_'),('Pure prediction','_prediction0_')]:
            rr=[r for r in token if key in r['method']];assert len(rr)==8
            best=max(rr,key=lambda r:r['mean_accuracy_pct'])
            text+=f"| {label} | {best['method']} | {best['mean_accuracy_pct']:.2f}% | {best['mean_preparation_updates_seconds']:.2f} | {min(r['mean_accuracy_pct'] for r in rr):.2f}–{best['mean_accuracy_pct']:.2f}% |\n"
        text+=f"\nWarmup accuracy at this start is {float(warm['accuracy']):.2f}%. Thus the best pure predictor shows a development improvement, but short Oracle32 reaches {short['mean_accuracy_pct']:.2f}% in {short['mean_preparation_updates_seconds']:.2f}s at only 32 updates. Predictor construction remains additional; a positive change from warmup does not establish a time-to-quality advantage.\n"
    behavior_path=EXP/'results/token_behavior_development_summary.json'
    if behavior_path.exists():
        behavior=json.loads(behavior_path.read_text());assert behavior['complete']
        text+='\n## Can the token critic predict the updated agent’s behavior?\n\n'
        text+='This separate development diagnostic uses 513 full prompts, three starts and five existing Oracle states per start. These are the existing **gradient-selected checkpoints**, including the distillation-only fit; they were not selected for behavior accuracy. It tests whether the same predictor can directly substitute for the changed agent, not the best achievable behavior distillation. It compares forced-format ABC predictions with the current teacher and with simply retaining the warmup teacher’s predictions. The same answer template is supplied independently of the gold label; causal and label-invariance checks passed. It does not measure free-generation behavior.\n\n'
        text+='| Critic | Oracle steps | Critic accuracy | Current teacher accuracy | Agreement with current teacher | Unchanged warmup agreement | Critic readout, s | Teacher readout, s |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in behavior['rows']:
            if r['steps'] not in [256,512]:continue
            text+=f"| {r['model']} | {r['steps']} | {r['accuracy_pct']:.2f}% | {r['teacher_accuracy_pct']:.2f}% | {r['teacher_agreement_pct']:.2f}% | {r['initial_teacher_agreement_pct']:.2f}% | {r['readout_seconds']:.2f} | {r['teacher_seconds']:.2f} |\n"
        text+='\nTimes are historical inference components averaged across starts; shared prefix preparation, model loading and offline fitting are additional. The first batch also includes the invariance checks. These development outcomes do not alter the frozen held-out comparison. [All 15 model/state summaries](results/token_behavior_development_summary.json).\n'
    optional=[r for r in result['groups'] if r['method'].startswith(('needed_','class_batched')) and r['steps']==256]
    if optional:
        text+='\n## Completion of previously registered controls\n\n'
        text+='These are single-start screens (301/401), not new held-out confirmations. “Needed” prepares frozen-prefix caches only for the fixed minibatch schedule, without consulting model outputs. “Class batched” packs class-stratified true-gradient samples together; it does not use a neural predictor. Times remain historical preparation/update components.\n\n'
        text+='| Control | Updates | Accuracy | Prep + updates, s |\n|---|---:|---:|---:|\n'
        for r in optional:
            text+=f"| {r['method']} | {r['steps']} | {r['mean_accuracy_pct']:.2f}% | {r['mean_preparation_updates_seconds']:.2f} |\n"
        memory=[r for r in result['rows'] if r['subset']=='candidate_0' and
                (r['evaluation'].startswith('optional_completed_') or r['evaluation']=='pilot_table')]
        if memory:
            text+='\nThe remaining minibatch gradient-memory grid uses candidate0 (1,024 training examples), start 301/401, and runs completed recipes through 1,024 updates. The following are **posthoc best development checkpoints among the completed recipes**, not independently selected or confirmed final methods. The budget cutoff left unequal learning-rate coverage; all completed checkpoint results and uncompleted recipe names remain in the inventories.\n\n'
            text+='| Family | Best observed recipe | Updates | Accuracy | Prep + updates, s |\n|---|---|---:|---:|---:|\n'
            for label,prefix in [('Oracle32','oracle32_'),('True-gradient subset','subset'),('Original CV','cv'),('Gradient memory','table')]:
                rr=[r for r in memory if r['method'].startswith(prefix)]
                if not rr:continue
                best=max(rr,key=lambda r:(r['accuracy'],-r['adaptation_seconds']))
                text+=f"| {label} | {best['method']} | {best['steps']} | {best['accuracy']:.2f}% | {best['adaptation_seconds']:.2f} |\n"
        status_path=EXP/'results/optional_completed_readout_inventory.json'
        if status_path.exists():
            status=json.loads(status_path.read_text())
            text+=f"\nRegistered optional recipes completed: **{len(status['completed_recipes'])}**; incomplete or unexecuted: **{len(status['uncompleted_recipes'])}**. Completed checkpoints with a full development readout: **{status['already_scored_entries']}**; still unscored: **{status['unscored_completed_entries']}**. The additional token-critic and memory-table grids, when completed, are included in the linked numerical inventory. An unexecuted or partial recipe is not a measured failure.\n"
    text+='\n[All late readouts, per-start differences and fitting receipts](results/late_development_summary.json). Other completed optional screens remain in the complete experiment inventory; missing readouts are not measured failures.\n'
    audit_path=EXP/'results/late_generated_records_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['late_summary_sha256']==sha(EXP/'results/late_development_summary.json')
        text+=f"\nThe [late-record audit](results/late_generated_records_audit.json) checked all **{audit['records']:,} generated records across {len(audit['entries'])} readouts**, including complete source prompts and recomputed grading.\n"
    (EXP/'report/LATE_DEVELOPMENT.md').write_text(text)
    log(dict(rendered=True,paired_policies=len(methods),kd_readouts=len(kd)))

if __name__=='__main__':main()
