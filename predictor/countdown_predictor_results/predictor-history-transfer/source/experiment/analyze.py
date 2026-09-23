"""Audit paired artifacts and summarize all predefined endpoints without selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (101, 102, 103)
METHODS = ('carry', 'reset', 'frozen', 'oracle')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return float(np.mean(values))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def bootstrap_questions(values):
    rng = np.random.default_rng(202609231)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    return [float(x) for x in np.quantile(values[draws].mean(1), [.025, .975])]


def summarize(rows, gradients=True):
    out = dict(pairs=len(rows), methods={}, seeds={})
    for method in METHODS:
        values = [r['methods'][method] for r in rows]
        entry = dict(ce_improvement=mean([v['ce_improvement'] for v in values]),
            expression_ce_improvement=mean([v['expression_ce_improvement'] for v in values]),
            ce_improved_pairs=sum(v['ce_improvement'] > 0 for v in values))
        if gradients and method != 'oracle':
            entry['heldout_gradient'] = {key: mean([v['heldout']['batch_vs_autograd'][key] for v in values])
                for key in ('relative_l2', 'cosine', 'norm_ratio')}
            entry['heldout_gradient']['per_example_relative_l2'] = mean([v['heldout']['lora']['relative_l2'] for v in values])
            if method in ('carry', 'reset'):
                entry['before_calibration'] = {key: mean([v['before_heldout']['batch_vs_autograd'][key] for v in values])
                    for key in ('relative_l2', 'cosine', 'norm_ratio')}
                entry['selected_epochs'] = [v['fit']['selected_epoch'] for v in values]
        out['methods'][method] = entry
    delta = [r['methods']['carry']['ce_improvement'] - r['methods']['reset']['ce_improvement'] for r in rows]
    out['carry_minus_reset_ce_improvement'] = mean(delta)
    out['carry_better_ce_pairs'] = sum(v > 0 for v in delta)
    out['carry_minus_reset_expression_ce_improvement'] = mean([
        r['methods']['carry']['expression_ce_improvement'] - r['methods']['reset']['expression_ce_improvement'] for r in rows])
    if gradients:
        diffs = [r['methods']['carry']['heldout']['batch_vs_autograd']['relative_l2'] -
                 r['methods']['reset']['heldout']['batch_vs_autograd']['relative_l2'] for r in rows]
        out['carry_minus_reset_gradient_relative_l2'] = mean(diffs)
        out['carry_better_gradient_pairs'] = sum(v < 0 for v in diffs)
        out['carry_better_cosine_pairs'] = sum(
            r['methods']['carry']['heldout']['batch_vs_autograd']['cosine'] >
            r['methods']['reset']['heldout']['batch_vs_autograd']['cosine'] for r in rows)
        out['relative_reduction_of_mean_error_percent'] = 100 * (1 -
            out['methods']['carry']['heldout_gradient']['relative_l2'] /
            out['methods']['reset']['heldout_gradient']['relative_l2'])
    for seed in SEEDS:
        subset = [r for r in rows if r['seed'] == seed]
        out['seeds'][str(seed)] = dict(
            carry_minus_reset_ce_improvement=mean([
                r['methods']['carry']['ce_improvement']-r['methods']['reset']['ce_improvement'] for r in subset]),
            methods={m: dict(ce_improvement=mean([r['methods'][m]['ce_improvement'] for r in subset]),
                **({'gradient_relative_l2': mean([r['methods'][m]['heldout']['batch_vs_autograd']['relative_l2'] for r in subset]),
                    'gradient_cosine': mean([r['methods'][m]['heldout']['batch_vs_autograd']['cosine'] for r in subset])}
                   if gradients and m != 'oracle' else {})) for m in METHODS})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-accuracy', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((HERE / 'data/manifest.json').read_text())
    assert sha(HERE / 'PLAN.md') == manifest['plan_sha256']
    seen = set()
    split_keys = {}
    for name, split in manifest['splits'].items():
        path = HERE / 'data' / f'{name}.jsonl'
        assert sha(path) == split['sha256']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        keys = {tuple(sorted(row['nums'])) for row in rows}
        assert len(rows) == len(keys) == split['n']
        assert not keys & seen
        assert sum(len(row['nums']) == 3 for row in rows) == len(rows)//2
        assert [r['source_index'] for r in rows] == split['source_indices']
        seen.update(keys)
        split_keys[name] = keys
    assert len(seen) == manifest['new_unique_number_multisets'] == 1784
    for path, identity in manifest['prior_files'].items():
        assert sha(path) == identity['sha256']
        prior = {tuple(sorted(row['nums'])) for row in
            (json.loads(line) for line in Path(path).read_text().splitlines()) if 'nums' in row}
        assert not prior & seen

    runs = []
    all_rows = []
    for seed in SEEDS:
        run = json.loads((HERE / 'results' / f'u{seed}.json').read_text())
        assert run['complete'] and run['frozen_base_unchanged']
        assert len(run['history']) == 6 and len(run['probes']) == 8
        assert run['manifest_sha256'] == sha(HERE / 'data/manifest.json')
        assert run['script_sha256'] == sha(HERE / 'run_seed.py')
        assert run['protocol']['input_truncation'] is False
        for h in run['history']:
            for m in ('carry', 'reset'):
                assert h[m]['optimizer_steps'] == 30
                assert len(h[m]['curve']) == 31
        assert run['history'][0]['carry']['selected_epoch'] == run['history'][0]['reset']['selected_epoch']
        for r in run['probes']:
            r['seed'] = seed
            assert r['starting_adapter_sha256'] == manifest['starts'][str(seed)]['model_states'][str(r['model_step'])]['sha256']
            for method, item in r['methods'].items():
                post = item['after_update']
                assert post['examples'] == len(post['rows']) == 256
                assert [v['source_index'] for v in post['rows']] == manifest['splits']['loss256']['source_indices']
                base = run['baselines'][str(r['model_step'])]
                assert abs(item['ce_improvement'] - (base['token_mean_ce']-post['token_mean_ce'])) < 1e-12
                if method in ('carry', 'reset'):
                    f = item['fit']
                    assert f['optimizer_steps'] == 30 and len(f['curve']) == 31
                    assert f['calibration_examples'] == 4 and f['selection_examples'] == 16
                    assert f['selected_score'] == min(x['selection_score'] for x in f['curve'])
            assert r['reconstruction_audit']['relative_l2'] < .05
            all_rows.append(r)
        runs.append(run)
    summary = dict(primary_model_step=32,
        main={str(step): summarize([r for r in all_rows if r['model_step'] == step]) for step in (16, 32)},
        audit=dict(complete_main_runs=3, probe_pairs=24, history_episodes_per_seed=6,
            new_data_examples=1784, prior_files_checked=len(manifest['prior_files']),
            no_new_split_overlap=True, no_prior_data_overlap=True,
            full_inputs_preserved=True, no_frozen_base_changes=True,
            first_episode_predictor_equality=True,
            max_autograd_reconstruction_relative_l2=max(r['reconstruction_audit']['relative_l2'] for r in all_rows)),
        limitations=['Three paired seeds and four shared probe batches per state are not 12 independent training trajectories.',
            'All model states follow true-gradient reference trajectories, not self-generated predictor rollouts.',
            'No overall compute saving or learned-learning-rate/meta-learning claim is established.'])

    flat_rows = []
    for row in all_rows:
        for method in METHODS:
            item = row['methods'][method]
            metric = item.get('heldout', {}).get('batch_vs_autograd', {})
            flat_rows.append(dict(seed=row['seed'], model_step=row['model_step'], probe=row['probe'], method=method,
                gradient_relative_l2=metric.get('relative_l2'), gradient_cosine=metric.get('cosine'),
                gradient_norm_ratio=metric.get('norm_ratio'), selected_epoch=item.get('fit', {}).get('selected_epoch'),
                ce_improvement=item['ce_improvement'], expression_ce_improvement=item['expression_ce_improvement']))
    with (HERE / 'results/paired_results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    controls = []
    for seed in SEEDS:
        path = HERE / 'results/low_lr' / f'u{seed}.json'
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        if not obj['complete']:
            continue
        assert obj['primary_result_sha256'] == sha(HERE / 'results' / f'u{seed}.json')
        assert obj['lr'] == 1e-4 and len(obj['probes']) == 8
        for row in obj['probes']:
            assert row['calibration_reproduced']
            row['seed'] = seed
            controls.append(row)
    if len(controls) == 24:
        summary['exploratory_low_lr'] = {str(step): summarize([r for r in controls if r['model_step'] == step], gradients=False)
                                       for step in (16, 32)}

    summary['costs'] = dict(
        reused_initial_predictor_training_examples=8192, reused_initial_selection_examples=256,
        primary_history_true_gradient_training_examples=24,
        primary_history_true_gradient_selection_examples=96,
        primary_history_predictor_optimizer_steps_per_method=180,
        current_true_gradient_training_examples=4, current_true_gradient_selection_examples=16,
        current_predictor_optimizer_steps=30, current_selection_forward_examples=31*16,
        primary_one_probe_lifetime_true_gradient_examples=140,
        primary_one_probe_lifetime_predictor_optimizer_steps=210,
        true_gradient_update_batch_examples=32, diagnostic_heldout_gradient_examples=28,
        mean_history_fit_seconds_per_seed={m:mean([sum(h[m]['seconds'] for h in r['history']) for r in runs]) for m in ('carry', 'reset')},
        mean_current_fit_seconds={m:mean([r['methods'][m]['fit']['seconds'] for r in all_rows]) for m in ('carry', 'reset')},
        mean_current_true_gradient_collection_seconds={k:mean([r['collection_seconds'][k] for r in all_rows])
            for k in ('calibration', 'selection', 'diagnostic_heldout')},
        caveat='History costs are paid by both sequential baselines; a reset-only one-off query could skip all history. Diagnostic/autograd and evaluation work is measured experiment overhead, not free deployment computation.')

    accuracy = {}
    arrays = {}
    output_diagnostics = {}
    reference_prompts = None
    for seed in SEEDS:
        root = HERE / 'results/accuracy' / f'u{seed}'
        path = root / 'summary.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        protocol = json.loads((root / 'protocol.json').read_text())
        assert protocol['input_truncation'] is False and protocol['max_new_output_tokens'] == 2048
        assert protocol['test_sha256'] == manifest['splits']['accuracy1024']['sha256']
        for method, entry in report.items():
            record_path = root / f'{method}.jsonl'
            assert sha(record_path) == entry['records_sha256']
            records = [json.loads(line) for line in record_path.read_text().splitlines()]
            assert len(records) == entry['n'] == 1024
            assert [r['source_index'] for r in records] == manifest['splits']['accuracy1024']['source_indices']
            prompts = [r['full_prompt'] for r in records]
            if reference_prompts is None:
                reference_prompts = prompts
            assert prompts == reference_prompts and all(prompts)
            values = np.array([int(r['correct']) for r in records], dtype=np.int8)
            assert int(values.sum()) == entry['correct']
            assert sum(r['finish_reason'] == 'length' for r in records) == entry['output_cap_hits']
            arrays[(seed, method)] = values
            output_diagnostics[(seed, method)] = dict(
                mean_output_tokens=mean([r['output_tokens'] for r in records]),
                correct_at_output_cap=sum(r['correct'] and r['finish_reason'] == 'length' for r in records))
        accuracy[str(seed)] = report
    summary['accuracy_evaluations_complete'] = len(arrays)
    if len(arrays) == 15:
        differences = np.stack([(arrays[(seed, 'carry')] - arrays[(seed, 'reset')]).astype(float) for seed in SEEDS])
        per_question = differences.mean(0)
        summary['accuracy'] = dict(by_seed=accuracy,
            mean_percent={m:100*mean([accuracy[str(s)][m]['accuracy'] for s in SEEDS]) for m in ('baseline',)+METHODS},
            mean_gain_pp={m:100*mean([accuracy[str(s)][m]['accuracy']-accuracy[str(s)]['baseline']['accuracy'] for s in SEEDS]) for m in METHODS},
            output_diagnostics={m:dict(
                mean_output_cap_percent=100*mean([accuracy[str(s)][m]['output_cap_hits']/1024 for s in SEEDS]),
                mean_output_tokens=mean([output_diagnostics[(s, m)]['mean_output_tokens'] for s in SEEDS]),
                correct_at_cap_across_runs=sum(output_diagnostics[(s, m)]['correct_at_output_cap'] for s in SEEDS))
                for m in ('baseline',)+METHODS},
            carry_minus_reset_pp=100*float(per_question.mean()),
            conditional_question_bootstrap95_pp=[100*v for v in bootstrap_questions(per_question)],
            uncertainty='Question-only bootstrap conditional on three fixed seeds and probe0. It excludes seed and calibration-batch uncertainty.',
            full_prompt_equality_verified=True, output_cap=2048,
            scope='Prespecified primary state32, probe0, LR3e-4 only.')
        comparisons = {}
        for reference in ('baseline', 'frozen', 'oracle'):
            difference = np.stack([(arrays[(s, 'carry')]-arrays[(s, reference)]).astype(float)
                                   for s in SEEDS]).mean(0)
            comparisons['carry_minus_' + reference] = dict(
                difference_pp=100*float(difference.mean()),
                conditional_question_bootstrap95_pp=[100*v for v in bootstrap_questions(difference)],
                descriptive_unadjusted=True)
        summary['accuracy']['supporting_contrasts'] = comparisons
    if args.require_accuracy:
        assert len(arrays) == 15
        assert len(controls) == 24
    write(HERE / 'results/analysis.json', summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(carry='#167c80', reset='#d58434', frozen='#969696', oracle='#6457a6')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, step in enumerate((16, 32)):
        for offset, method in enumerate(('frozen', 'reset', 'carry')):
            values = [r['methods'][method]['heldout']['batch_vs_autograd']['relative_l2'] for r in all_rows if r['model_step'] == step]
            x = i+(offset-1)*.23
            axes[0].bar(x, mean(values), width=.21, color=colors[method], label=method if i == 0 else None)
            axes[0].scatter(np.linspace(x-.06, x+.06, len(values)), values, s=12, color='black', alpha=.35)
        for offset, method in enumerate(METHODS):
            x = i+(offset-1.5)*.19
            axes[1].bar(x, 1000*summary['main'][str(step)]['methods'][method]['ce_improvement'], width=.18,
                        color=colors[method], label=method if i == 0 else None)
            if 'exploratory_low_lr' in summary:
                axes[2].bar(x, 1000*summary['exploratory_low_lr'][str(step)]['methods'][method]['ce_improvement'],
                            width=.18, color=colors[method], label=method if i == 0 else None)
    axes[0].set_ylabel('Held-out gradient relative L2 (lower better)')
    axes[0].set_title('Gradient transfer: 4 probes x 3 seeds')
    for ax, title in zip(axes[1:], ('Main update LR=3e-4', 'Exploratory update LR=1e-4')):
        ax.set_title(title)
        ax.set_ylabel('Completion CE improvement x 1000')
        ax.axhline(0, color='black', linewidth=.8)
    for ax in axes:
        ax.set_xticks([0, 1], ['Model step 16', 'Model step 32'])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(HERE / 'results/history_transfer.png', dpi=180)
    plt.close(fig)
    if 'accuracy' in summary:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        method_order = ('baseline',) + METHODS
        x = np.arange(len(method_order))
        for seed in SEEDS:
            ax.plot(x, [100*accuracy[str(seed)][m]['accuracy'] for m in method_order],
                    marker='o', alpha=.6, linewidth=1, label=f'Paired seed {seed}')
        ax.plot(x, [summary['accuracy']['mean_percent'][m] for m in method_order],
                marker='D', color='black', linewidth=2, label='Mean of 3 seeds')
        ax.set_xticks(x, ['Before update', 'Carry history', 'Reset', 'Frozen', 'True gradient'])
        ax.set_ylabel('Exact Countdown accuracy (%)')
        ax.set_title('Prespecified state32 / probe0 / 1024 shared questions')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / 'results/accuracy_comparison.png', dpi=180)
        plt.close(fig)
    print(json.dumps(dict(main={k:{key:v[key] for key in ('carry_better_gradient_pairs','carry_better_ce_pairs',
        'carry_minus_reset_ce_improvement','relative_reduction_of_mean_error_percent')} for k,v in summary['main'].items()},
        complete_accuracy=len(arrays), low_lr_pairs=len(controls)), indent=2))


if __name__ == '__main__':
    main()
