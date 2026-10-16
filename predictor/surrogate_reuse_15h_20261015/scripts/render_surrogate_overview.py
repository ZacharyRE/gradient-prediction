"""Build the concise English overview only from completed frozen evidence."""
from common import *


def one(rows, **wanted):
    found = [r for r in rows if all(r.get(k) == v for k, v in wanted.items())]
    assert len(found) == 1, wanted
    return found[0]


def pp(metric):
    return f"{100*metric['mean']:+.2f} pp [{100*metric['ci95'][0]:+.2f}, {100*metric['ci95'][1]:+.2f}]"


def main():
    audit = json.loads((EXP/'results/final_evidence_audit.json').read_text())
    assert audit['passed']
    analysis = json.loads((EXP/'results/final_selection_analysis.json').read_text())
    cost = json.loads((EXP/'results/surrogate_cost_comparison.json').read_text())
    assert sha(EXP/'results/surrogate_cost_comparison.json') == audit['evidence_sha256']['results/surrogate_cost_comparison.json']
    workflow = cost['measured_selection_pipeline']
    assert workflow['complete']
    entries = [
        ('Short Oracle32', 'oracle32_matched', 128),
        ('Primary Oracle32', 'oracle32_matched', 256),
        ('Stronger Oracle32, graph', 'graph_oracle32_lr0.0003', 256),
        ('Matched Oracle16', 'oracle16_matched', 256),
        ('Original CV16', 'cv16_matched', 256),
        ('State CV16 first 32, then Oracle16', 'state256_cv16_until32_lr0.001', 256),
        ('Tuned original CV16', 'cv16_lr0.0003', 512),
        ('Tuned Oracle16', 'subset16_lr0.0003', 512),
        ('Original CV8, graph', 'graph_cv8_lr0.001', 512),
        ('Matched Oracle8, graph', 'graph_subset8_lr0.001', 512),
        ('Zero / Adam, true gradient every 4', 'zero_every4_lr0.001', 256),
    ]
    first = one(analysis['direct_quality'], family='fullpool', method='cv16_matched', steps=256)
    warmup = 100*(first['accuracy']['mean'] - first['delta_vs_warmup']['mean'])
    table = ['| Method | Updates | LR | Accuracy % | Prep + updates, s | Fresh process, s |',
             '|---|---:|---:|---:|---:|---:|',
             f'| Warmup only | 0 | — | {warmup:.2f} | 0 incremental | 0 incremental |']
    for label, method, steps in entries:
        row = one(cost['direct_training'], family='fullpool', method=method, steps=steps)
        assert row['qualified_all_starts']
        table.append(f"| {label} | {steps} | {row['lr']:g} | {row['accuracy_pct']:.2f} | {row['qualified_mean_preparation_plus_online_seconds']:.2f} | {row['qualified_mean_fresh_script_seconds']:.2f} |")
    def pair(actual, reference):
        found = [r for r in analysis['direct_paired_comparisons']
                 if r['family']=='fullpool' and r['actual']['method']==actual
                 and r['reference']['method']==reference]
        assert len(found)==1
        return pp(found[0]['delta'])
    early_old = pair('state256_cv16_until32_lr0.001', 'cv16_matched')
    early_matched = pair('state256_cv16_until32_lr0.001', 'oracle16_matched')
    early_strong = pair('state256_cv16_until32_lr0.001', 'graph_oracle32_lr0.0003')
    original_offline = one(cost['direct_training'], family='fullpool', method='cv16_matched', steps=256)['offline']['recorded_component_sum_seconds']
    state_offline = one(cost['direct_training'], family='fullpool', method='state256_cv16_until32_lr0.001', steps=256)['offline']['recorded_component_sum_seconds']
    cv_candidate = one(cost['direct_training'], family='random', method='cv16_matched', steps=128)
    short_candidate = one(cost['direct_training'], family='random', method='oracle32_matched', steps=64)
    assert all(r['qualified_all_starts'] and len(r['profiled_matrix_flops']) == 1
               for r in [cv_candidate, short_candidate])
    cv_pflops = cv_candidate['profiled_matrix_flops'][0]['flops']/1e15
    short_pflops = short_candidate['profiled_matrix_flops'][0]['flops']/1e15
    def selection(family, method, selector, steps=128):
        return one(analysis['groups'], family=family, method=method, steps=steps,
                   target='all', selector=selector)
    selectors = [
        ('Oracle / generated accuracy', 'oracle32_lr0.0003', 'generated_accuracy', 128),
        ('Oracle / exact teacher probability', 'oracle32_lr0.0003', 'exact_teacher_gold_probability', 128),
        ('Short Oracle / exact teacher probability', 'oracle32_matched', 'exact_teacher_gold_probability', 64),
        ('CV16 / generated accuracy', 'cv16_matched', 'generated_accuracy', 128),
        ('CV16 / exact teacher probability', 'cv16_matched', 'exact_teacher_gold_probability', 128),
        ('CV16 / learned behavior probability', 'cv16_matched', 'mnli_behavior_anchored128_coverage:gold_probability', 128),
    ]
    selection_table = ['| Screening method | Random candidates % | Genre candidates % |',
                       '|---|---:|---:|']
    for label, method, selector, steps in selectors:
        values = [100*selection(f,method,selector,steps)['selection_oracle_accuracy']['mean'] for f in ['random','genre']]
        selection_table.append(f'| {label} | {values[0]:.2f} | {values[1]:.2f} |')
    cvrows = [selection(f,'cv16_matched','generated_accuracy') for f in ['random','genre']]
    random_quality = [100*(r['selection_oracle_accuracy']['mean']-r['selection_delta_vs_random']['mean']) for r in cvrows]
    selection_table.append(f'| Random choice, expectation | {random_quality[0]:.2f} | {random_quality[1]:.2f} |')
    names = {
        'oracle_generated':'Oracle32 + generation', 'oracle_forced':'Oracle32 + exact ABC score',
        'cv16_generated':'CV16 + generation', 'cv16_forced':'CV16 + exact ABC score',
        'cv16_behavior':'CV16 + learned behavior score',
        'short_oracle_forced':'Short Oracle32 + exact ABC score',
    }
    workflow_table = ['| Complete workflow | Median seconds | Offline construction components, s |', '|---|---:|---:|']
    for row in workflow['rows']:
        assert row['three_repeats_complete']
        construction = one(cost['selection_aggregates'], family=row['family'],
                           method=row['method'], steps=row['steps'], target='all',
                           selector=row['selector'])['recorded_offline_component_sum_seconds']
        workflow_table.append(f"| {names[row['condition']]} | {row['pipeline_seconds_median']:.2f} | {construction:.1f} |")
    fidelity = json.loads((EXP/'results/final_behavior_fidelity.json').read_text())
    f = one(fidelity['rows'], family='fullpool', method='oracle32_matched', steps=256)['metrics']
    text = f'''# LoRA gradient predictors as reusable training surrogates

**Hybrid predictor updates can approach Oracle accuracy, but this study does not establish an overall predictor efficiency win.** Shorter Oracle training and cheap true-gradient controls remain competitive. The separate behavior surrogate does not yet capture the trained agent's improvement.

## Direct training

Qwen2.5-1.5B-Instruct, MNLI, one H200. Generated accuracy averages **three new starts on 2,049 heldout examples**. Updates are additional to a shared 32-update warmup. Each time averages the three starts' medians from **three exact-weight timing repetitions**.

{chr(10).join(table)}

Times exclude shared target warmup, evaluation and offline predictor construction. Fresh-process time additionally includes imports/loading and final save. Recorded offline construction is **{original_offline:.1f}s** for the original predictor and **{state_offline:.1f}s** for State256, including source warmups/dependencies; these are historical component totals, not clean cold-start benchmarks. Reuse spreads construction cost but does not remove per-update predictor overhead. [All horizons, matrix FLOPs, costs and intervals](RESULTS.md).

**Fewer FLOPs need not mean less time:** on representative candidate0, CV16/128 costs **{cv_pflops:.2f} matrix PFLOPs / {cv_candidate['qualified_mean_preparation_plus_online_seconds']:.2f}s**, versus short Oracle32/64's **{short_pflops:.2f} PFLOPs / {short_candidate['qualified_mean_preparation_plus_online_seconds']:.2f}s**. FLOPs use start 302; times average all three starts' medians.

The early state-predictor recipe improves over original CV16/256 by **{early_old}**, but its difference from matched Oracle16 is **{early_matched}** and from the stronger graph Oracle32 is **{early_strong}**. These paired 95% intervals do not establish superiority or noninferiority to Oracle. The recipe changes both fitting and the duration of predictor use.

## Selecting training data

Screen five 1,024-example datasets on selection-dev, then deploy the **reference Oracle trained on the chosen dataset**. These are its heldout accuracies, not the simulator adapter's accuracy. Simulations use 128 updates, except short Oracle's 64; the reference is Oracle32/128/LR 0.0003.

{chr(10).join(selection_table)}

CV16/generated scoring gains **{pp(cvrows[0]['selection_delta_vs_random'])}** over random choice on random candidates and **{pp(cvrows[1]['selection_delta_vs_random'])}** on genre candidates. Neither establishes a selection advantage over random choice.

The actual complete-workflow benchmark below uses **start 302, five random candidates and three timing repeats**. It charges fresh adaptations, scoring, process transitions/loading and final Oracle training when needed; exhaustive Oracle selection reuses its chosen adapter. Offline construction and common warmup remain additional.

{chr(10).join(workflow_table)}

This implementation uses separate processes, not a persistent model server. Timing repeats are not independent scientific starts. Keeping the selected simulator adapter avoids final Oracle training, but changes the deployment endpoint; those frozen secondary quality results and component costs are separate in [RESULTS](RESULTS.md).

## What the surrogate learned

For the primary Oracle32/256 agent, forced-format ABC accuracy is **{100*f['teacher_accuracy']['mean']:.2f}%**; the learned behavior model reaches **{100*f['surrogate_accuracy']['mean']:.2f}%**, versus **{100*f['initial_accuracy']['mean']:.2f}%** for unchanged initial predictions. Its agreement gain with the changed agent is **{pp(f['agreement_delta_vs_initial'])}**. Cheap inference alone has not produced a reliable behavior replacement.

Late-state gradient correlation also weakens. State conditioning, additional state coverage, mean-aware fitting, sparse calibration, scalar/token critics, copied teacher blocks, gradient memory and local linearization were tested during development. [Attempt-by-attempt evidence and failures](DEVELOPMENT.md).

![Frozen quality and cost tradeoffs; selection panels use component estimates](results/figures/frozen_surrogate_tradeoffs.png)

## Setup and interpretation

- Only decoder layer index 8 (ninth block), attention output-projection LoRA is trained: rank/alpha 64; fresh AdamW state after warmup, batch 32, clipping 1, no weight decay.
- Final warmup/update seeds: 302/401, 303/402, 304/403. Fullpool adaptation uses 4,097 examples. Five random candidates may overlap; five genre candidates use disjoint source groups.
- Selection-dev: 1,026 examples; holdout: 2,049. Recipes precede selection-dev results; choices precede holdout results. These are **MNLI training-population holdouts**, not the official validation benchmark. No new 0.5B, 7B or PAWS result is claimed.
- All evaluation inputs are preserved. Greedy generation permits at most 256 **new output tokens**. Forced ABC scores are a separate metric.
- **Oracle16/8** uses a true-gradient subset without a predictor. **CV16/8** predicts all 32 gradients and corrects with 16/8 true gradients. State256 names predictor width; its early-use recipe switches to Oracle16 after 32 updates. [Method definitions](METHODS.md).

**Next priority:** beat short Oracle at equal total time. For gradient prediction, first improve late-state correlation on new starts. For Ben's data-selection use case, test a model that predicts which dataset yields a larger improvement; reconstructing every changed-agent logit may be unnecessary. That ranking model remains future work and must beat random selection and cheap Oracle controls with its construction cost charged.

[Protocol](PROTOCOL.md) · [Literature](LITERATURE.md) · [Reproduction](REPRODUCING.md) · [Complete numerical results](RESULTS.md)
'''
    (EXP/'report/README.md').write_text(text)
    write(EXP/'provenance/overview_render.json', dict(epoch=time.time(),
          final_evidence_audit_sha256=sha(EXP/'results/final_evidence_audit.json'),
          readme_sha256=sha(EXP/'report/README.md'), words=len(text.split())))
    log(dict(rendered=True, words=len(text.split())))


if __name__ == '__main__':
    main()
