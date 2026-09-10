"""Freeze a deliberately selected recipe file before any final benchmark output.

This is a CPU-only preparation tool. It never selects a recipe or launches evaluation.
Selection JSON: primary_family, families ({family: {seed: adapter_name}}),
selection_reason, and control_rationale. All declared families use all five seeds.
"""
import argparse
from common import *
from gradient_geometry.sft_protocol import artifact_identity


def provenance(adapter, seed):
    ready = json.loads((adapter/'ready.json').read_text())
    manifest_path = adapter.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    result = dict(adapter=str(adapter), ready=ready, manifest=manifest,
                  manifest_sha256=sha(manifest_path), files=artifact_identity(adapter)['files'])
    if manifest.get('kind') == 'derived_epoch_weight_average_not_new_training':
        result['training_sources'] = []
        signatures = []
        for item in manifest['sources']:
            source = Path(item['path'])
            assert sha(source/'adapter_model.safetensors') == item['weight_sha256']
            assert sha(source/'adapter_config.json') == item['config_sha256']
            evidence, signature = provenance(source, seed)
            result['training_sources'].append(evidence)
            signatures.append(signature)
        # The frozen endpoint sequence must match across seeds too.
        signature = dict(kind=manifest['kind'], sources=signatures,
                         inference_rank=manifest['inference_rank'], formula=manifest['formula'])
    else:
        arguments = dict(manifest['arguments'])
        source_path = adapter.parent/'train_source.py'
        assert sha(source_path) == manifest['script_sha']
        assert not manifest.get('resumed_from') and arguments.get('resume_from') is None
        replay_path = S/'audits/anchor_zero_replay_result.json'
        replay = json.loads(replay_path.read_text())
        assert replay['passed'] and replay['exposure_sequence_equal'] and replay['history_except_timestamps_equal']
        assert sha(S/'scripts/train.py') == replay['current_original_source_sha256']
        assert sha(S/'audits/anchor_zero_legacy_source_diff.txt') == replay['source_diff_sha256']
        standard_versions = [replay['original_training_source_sha256'], replay['current_original_source_sha256']]
        if manifest['script_sha'] in standard_versions:
            code_signature = dict(kind='standard_fresh_training_with_audited_optional_resume_extension',
                                  source_versions=standard_versions, replay_sha256=sha(replay_path))
        else:
            assert manifest['script_sha'] == sha(S/'scripts/train_anchored.py')
            assert manifest['anchored_helper_sha256'] == sha(S/'scripts/anchored_objective.py')
            assert manifest['forked_original_train_sha256'] == sha(S/'scripts/train.py')
            code_signature = dict(kind='current_anchored_trainer', source_sha256=manifest['script_sha'],
                                  helper_sha256=manifest['anchored_helper_sha256'])
        assert arguments.pop('seed') == seed
        arguments.pop('name')
        # Optional resume support was introduced after the original seed43 run.
        arguments.setdefault('resume_from', None)
        train_file = Path(arguments['train_file'])
        assert sha(train_file) == manifest['data_sha']
        signature = dict(arguments=arguments, data_sha256=manifest['data_sha'],
                         examples=manifest['examples'], model=manifest['model'],
                         training_code=code_signature,
                         common_source_sha256=sha(adapter.parent/'common_source.py'),
                         endpoint=ready['epoch'])
    return result, signature


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', type=Path, required=True)
    a = p.parse_args()
    final = S/'results/confirmation_plan.json'
    assert not final.exists(), 'Confirmation plan is already frozen; never overwrite it.'
    selection = json.loads(a.selection.read_text())
    native_plan = json.loads((S/'audits/final_native_protocol_plan.json').read_text())
    assert native_plan['n'] == 500 and native_plan['batch'] == 8
    assert native_plan['data_sha256'] == sha(S/'data/dev.jsonl')
    assert native_plan['native_script_sha256'] == sha(S/'scripts/native_final_recheck.py')
    holdout_path = S/'audits/math_without_current_validation_plan.json'
    holdout = json.loads(holdout_path.read_text())
    assert holdout['data_sha256'] == sha(S/'data/math_reused5000.jsonl')
    assert holdout['validation_sha256'] == sha(S/'data/math_reused1500.jsonl')
    assert len(holdout['keep_indices']) == len(set(holdout['keep_indices'])) == 3500
    validation = read(S/'data/math_reused1500.jsonl')
    full_math = read(S/'data/math_reused5000.jsonl')
    validation_indices = [row['source_test_index'] for row in validation]
    assert len(validation_indices) == len(set(validation_indices)) == 1500
    assert holdout['keep_indices'] == [i for i in range(len(full_math))
                                      if i not in set(validation_indices)]
    for row, index in zip(validation, validation_indices):
        assert row['problem'] == full_math[index]['problem']
        assert row['solution'] == full_math[index]['solution']
    seeds = [43, 44, 45, 46, 47]
    assert selection['selection_reason'] and 'control_rationale' in selection
    families = selection['families']
    assert selection['primary_family'] in families
    datasets = ['math_reused5000', 'ood_minerva', 'ood_olympiad', 'ood_svamp', 'ood_amc23']
    for dataset in datasets:
        root = S/'results/evaluation'/dataset
        assert not list(root.glob('*/math_predictions.jsonl')), f'{dataset}: final outputs already exist'
        assert not list(root.glob('*/math_summary.json')), f'{dataset}: final scores already exist'
    replay_path = S/'audits/sampling_restart_protocol_replay.json'
    assert json.loads(replay_path.read_text())['passed']
    if (S/'audits/teacher_five_seed_math1500_extension_plan.json').exists():
        replay_path = S/'audits/teacher1500_restart_protocol_replay.json'
        assert json.loads(replay_path.read_text())['passed']
    if (S/'audits/coverage_math1500_frozen.json').exists():
        replay_path = S/'audits/coverage1500_restart_protocol_replay.json'
        assert json.loads(replay_path.read_text())['passed']
    contract_path = S/'audits/final_data_contract.json'
    contract = json.loads(contract_path.read_text())
    assert contract['passed']
    numeric_path = S/'audits/minerva_numeric_contract.json'
    numeric = json.loads(numeric_path.read_text())
    assert numeric['passed'] and numeric['n_numeric'] == 191 and numeric['n_symbolic'] == 81
    assert numeric['dataset_sha256'] == sha(S/'data/ood_minerva.jsonl')
    assert numeric['helper_sha256'] == sha(S/'scripts/minerva_numeric.py')
    amendment_path = S/'audits/minerva_numeric_scoring_amendment.json'
    amendment = json.loads(amendment_path.read_text())
    assert amendment['no_final_outputs_at_declaration']
    assert amendment['contract_sha256'] == sha(numeric_path)
    assert amendment['scorer_sha256'] == sha(S/'scripts/score_minerva_numeric.py')
    numeric_integration_path = S/'audits/minerva_scorer_integration.json'
    numeric_integration = json.loads(numeric_integration_path.read_text())
    assert numeric_integration['passed']
    assert numeric_integration['scorer_sha256'] == sha(S/'scripts/score_minerva_numeric.py')
    assert numeric_integration['helper_sha256'] == sha(S/'scripts/minerva_numeric.py')
    analysis_integration_path = S/'audits/final_analysis_integration.json'
    analysis_integration = json.loads(analysis_integration_path.read_text())
    assert analysis_integration['passed'] and analysis_integration['analysis_script_sha256'] == sha(S/'scripts/analyze_confirmation.py')
    numeric_math_path = S/'audits/math_numeric_grader_sensitivity_plan.json'
    numeric_math = json.loads(numeric_math_path.read_text())
    assert numeric_math['primary_unchanged'] and numeric_math['dataset_sha256'] == sha(S/'data/math_reused5000.jsonl')
    assert numeric_math['audit_sha256'] == sha(S/'audits/math_numeric_negative_controls.json')
    for dataset in datasets:
        assert contract['datasets'][dataset]['dataset_sha256'] == sha(S/f'data/{dataset}.jsonl')
    replay = json.loads(replay_path.read_text())
    baseline_replays = [r for r in replay['comparisons'] if r['reference'] == 'base']
    assert len(baseline_replays) == 1
    baseline_name = baseline_replays[0]['new']
    baseline_root = S/'results/evaluation/dev'/baseline_name
    assert sha(baseline_root/'math_predictions.jsonl') == baseline_replays[0]['new_predictions_sha256']
    baseline_path = baseline_root/'math_manifest.json'
    baseline = json.loads(baseline_path.read_text())
    assert sha(S/'scripts/evaluate.py') == baseline['script_sha256']
    bridge_path = S/'audits/evaluator_version_bridge.json'
    bridge = json.loads(bridge_path.read_text())
    assert bridge['passed'] and bridge['current_script_sha256'] == baseline['script_sha256']
    assert artifact_identity(Path(MODEL)) == baseline['model']
    ready = {}
    for path in (S/'results/training').glob('*/epoch*/ready.json'):
        row = json.loads(path.read_text())
        assert row['name'] not in ready
        ready[row['name']] = Path(row['adapter'])
    models = {}; artifacts = {}; signatures = {}
    for family, mapping in families.items():
        assert set(mapping) == {str(seed) for seed in seeds}
        signatures[family] = []
        for seed in seeds:
            name = mapping[str(seed)]
            assert name in ready
            adapter = ready[name]
            evidence, signature = provenance(adapter, seed)
            signatures[family].append(signature)
            models[name] = str(adapter)
            artifacts[name] = evidence
        assert all(s == signatures[family][0] for s in signatures[family]), family
    reserved = json.loads((S/'audits/ood_reserved.json').read_text())
    for dataset in datasets[1:]:
        assert sha(S/f'data/{dataset}.jsonl') == reserved['datasets'][dataset[4:]]['sha256']
    plan = dict(schema=1, frozen_at=time.time(), selection=selection,
                selection_file_sha256=sha(a.selection), seeds=seeds, families=families,
                primary_family=selection['primary_family'], primary_dataset=datasets[0],
                datasets=datasets, macro_datasets=datasets[1:4],
                sensitivity_exclusions={'math_reused5000': [2121, 2432, 4678]},
                validation_excluded_sensitivity=holdout,
                validation_excluded_plan_sha256=sha(holdout_path),
                data_sha256={d: sha(S/f'data/{d}.jsonl') for d in datasets},
                evaluation_protocol={k: baseline[k] for k in ['schema', 'script_sha256', 'engine',
                    'batch_invariant', 'max_tokens', 'temperature', 'chunk_size']},
                base_model_files_sha256=baseline['model']['files'],
                adapter_sha256={n: r['files']['adapter_model.safetensors'] for n, r in artifacts.items()},
                adapter_files_sha256={n: r['files'] for n, r in artifacts.items()},
                artifacts=artifacts, recipe_signatures={f: x[0] for f, x in signatures.items()},
                baseline_manifest_sha256=sha(baseline_path), restart_replay_sha256=sha(replay_path),
                baseline_protocol_reference=baseline_name,
                evaluator_version_bridge_sha256=sha(bridge_path),
                training_zero_replay_sha256=sha(S/'audits/anchor_zero_replay_result.json'),
                allowed_development_script_sha256=bridge['allowed_development_script_sha256'],
                final_data_contract_sha256=sha(contract_path),
                analysis_script_sha256=sha(S/'scripts/analyze_confirmation.py'),
                final_pipeline_script_sha256=sha(S/'scripts/final_evaluation_pipeline.py'),
                native_protocol_plan_sha256=sha(S/'audits/final_native_protocol_plan.json'),
                math_replay_script_sha256=sha(S/'scripts/audit_final_math_replay.py'),
                native_analysis_script_sha256=sha(S/'scripts/analyze_final_native.py'),
                native_strict_prepare_script_sha256=sha(S/'scripts/prepare_final_native_strict.py'),
                final_collector_script_sha256=sha(S/'scripts/collect_final_evidence.py'),
                strict_scoring_script_sha256=sha(S/'scripts/score_sensitivity.py'),
                minerva_numeric_contract_sha256=sha(numeric_path),
                minerva_numeric_amendment_sha256=sha(amendment_path),
                minerva_numeric_helper_sha256=sha(S/'scripts/minerva_numeric.py'),
                minerva_numeric_scoring_script_sha256=sha(S/'scripts/score_minerva_numeric.py'),
                minerva_numeric_integration_audit_sha256=sha(numeric_integration_path),
                analysis_integration_audit_sha256=sha(analysis_integration_path),
                math_numeric_grader_sensitivity=numeric_math,
                math_numeric_grader_sensitivity_plan_sha256=sha(numeric_math_path),
                scope='MATH5000 is historically reused. Only reserved mathematical OOD is fresh. '
                      'Five seeds and positive question/seed-question intervals are required; '
                      'no general nonmathematical capability or universal seed claim.',
                script_sha256=sha(__file__))
    write(S/'results/confirmation_manifest.json', {'base': None, **models})
    plan['evaluation_manifest_sha256'] = sha(S/'results/confirmation_manifest.json')
    write(final, plan)
    print(json.dumps(dict(frozen=str(final), families=list(families), models=len(models), seeds=seeds)))


if __name__ == '__main__':
    main()
