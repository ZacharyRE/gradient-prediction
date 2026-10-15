"""Render numerical frozen results; interpretation remains in the reviewed README."""
from common import *


def number(x,digits=2):
    return '—' if x is None else f'{x:.{digits}f}'


def main():
    result=json.loads((EXP/'results/final_comparison.json').read_text())
    plan=json.loads((EXP/'configs/final_selection_plan.json').read_text())
    labels={(r['task'],r['cohort']):r.get('label',r['cohort'].replace('_',' ')) for r in plan['cohorts']}
    lines=['# Frozen results','',
        'Recipes and stopping points were selected on development data and frozen before these readouts. Each trained cohort uses update seeds 101/102/103; warmup is one shared starting adapter. These evaluation populations were used in earlier studies, so the evidence is exploratory.','',
        'Accuracy intervals and paired gaps resample update seeds and whole source groups. The practical margin is −1 percentage point; a point estimate within that margin alone does not establish noninferiority.','']
    for task in ['paws','mnli']:
        rows=[r for r in result['rows'] if r['task']==task]
        lines += [f'## {task.upper()}','',
            '| Method | Steps | LR | Test accuracy % [95% CI] | Gap to Oracle, pp [95% CI] |',
            '|---|---:|---:|---:|---:|']
        for r in rows:
            label=labels.get((task,r['cohort']),'Warmup')
            if r['primary_candidate']:label+=' (primary predictor)'
            accuracy=f"{number(r['accuracy_pct'])} [{number(r['accuracy_ci95_low_pct'])}, {number(r['accuracy_ci95_high_pct'])}]"
            gap='Reference' if r['cohort']==r['reference'] else f"{number(r['gap_pp'])} [{number(r['gap_ci95_low_pp'])}, {number(r['gap_ci95_high_pp'])}]"
            lines.append(f"| {label} | {r['update_steps']} | {r['lr']:g} | {accuracy} | {gap} |")
        lines += ['', '| Method | Online + preparation, s | Standalone script, s [range] | Recorded predictor preparation, s | Matrix PFLOPs incl. cache | Replay weights match? |',
            '|---|---:|---:|---:|---:|---|']
        for r in rows:
            if r['cohort']=='warmup':continue
            label=labels[(task,r['cohort'])]
            standalone=f"{number(r['standalone_script_seconds_median'])} [{number(r['standalone_script_seconds_min'])}, {number(r['standalone_script_seconds_max'])}]"
            match='Yes, all 3 registered seeds' if r['eligible_for_direct_frozen_quality_speed_claim'] else 'Not established; see receipts'
            lines.append(f"| {label} | {number(r['online_plus_preparation_seconds_median'])} | {standalone} | {number(r['inherited_predictor_preparation_seconds_mean'])} | {number(r['total_matrix_pflops'],3)} | {match} |")
        lines.append('')
    confirmation=json.loads((EXP/'results/frozen_oracle_confirmation_analysis.json').read_text())
    gaps={(r['task'],r['cohort']):r for r in confirmation['comparisons']}
    lines += ['## Confirmation readout','',
        'All frozen cohorts are also evaluated on the 512-example confirmation split for each task. These smaller readouts have wider uncertainty; no recipe was changed after seeing them.','',
        '| Task | Method | Accuracy % | Gap to selected Oracle32, pp [95% CI] |',
        '|---|---|---:|---:|']
    for r in confirmation['results']:
        g=gaps.get((r['task'],r['cohort']))
        gap='Reference' if g is None else f"{number(g['mean_gap_pp'])} [{number(g['ci95_gap_pp'][0])}, {number(g['ci95_gap_pp'][1])}]"
        lines.append(f"| {r['task'].upper()} | {labels.get((r['task'],r['cohort']),'Warmup')} | {number(r['mean_pct'])} | {gap} |")
    lines += ['', 'Times are medians of three clean standalone replays. Online + preparation includes adaptation and required cache/initialization; standalone additionally includes imports, loading, and final serialization. Both exclude inherited predictor preparation, shared warmup training, and evaluation. Recorded predictor preparation is a separate component mean, not a newly timed cold pipeline. A dash means unmeasured.','',
        'FLOPs are seed-101 matrix-operation counts for the matching execution module, including cache preparation but excluding inherited predictor fitting and scalar/nonlinear/optimizer arithmetic. The small RLS linear solves are also uncounted; each receipt lists operator coverage. Instrumented runtimes are not used as speed measurements.','',
        'Machine-readable values: [CSV](results/final_comparison.csv), [test accuracy and paired intervals](results/frozen_test_analysis.json), [confirmation analysis](results/frozen_oracle_confirmation_analysis.json), [supplementary paired controls](results/frozen_control_comparisons.json), [timing receipts](results/frozen_timing_analysis.json), [preparation ledger](results/predictor_preparation_costs.json).','']
    (EXP/'report/RESULTS.md').write_text('\n'.join(lines))
    log(dict(rendered_rows=len(result['rows'])))


if __name__=='__main__':main()
