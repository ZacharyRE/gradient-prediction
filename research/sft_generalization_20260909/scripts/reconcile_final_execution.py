"""Reconcile intentional CPU-dispatch retirement with actual completed evidence.

Read only the original execution records; never replace their nonzero returncodes.
Run after both replacement benchmark execution and original native/collector finish.
"""
from common import *

sources = {}

def get(relative):
    path = S/relative
    value = json.loads(path.read_text())
    sources[relative] = sha(path)
    return value

amendment_path = ('audits/balanced_ood_zombie_read_amendment.json'
    if (S/'audits/balanced_ood_zombie_read_amendment.json').exists()
    else 'audits/balanced_ood_execution_runtime_amendment.json')
amend = get(amendment_path)
if amendment_path.endswith('zombie_read_amendment.json'):
    previous = get('audits/balanced_ood_execution_runtime_amendment.json')
    assert amend['prior_runtime_amendment_sha256'] == sources['audits/balanced_ood_execution_runtime_amendment.json']
    reader = get('audits/balanced_ood_zombie_read_contract.json')
    assert reader['passed'] and reader['new_script_sha256'] == amend['execution_script_sha256']
    assert amend['process_reader_contract_sha256'] == sources['audits/balanced_ood_zombie_read_contract.json']
    prior_status = get(amend['resume_state_path'])
    assert prior_status['status'] == 'error' and not prior_status['running'] and not prior_status['calls']
    assert all(t['status'] == 'pending' for t in prior_status['tasks'])
    assert sources[amend['resume_state_path']] == amend['resume_state_sha256']
plan = get('results/confirmation_plan.json')
ph = sources['results/confirmation_plan.json']
assert ph == amend['primary_plan_sha256']
for filename, digest in amend['unchanged_scripts'].items():
    assert sha(S/'scripts'/filename) == digest, filename
status = get('audits/balanced_ood_execution.json')
assert status['status'] in ['all_gpu_evaluations_complete', 'incomplete_error_or_cutoff']
assert status['main_gpu_complete'] and not status['running']
assert status['execution_amendment_sha256'] == sources[amendment_path]
assert status['script_sha256'] == amend['execution_script_sha256']
completion = get('audits/final_campaign_completion.json')
assert completion['plan_sha256'] == ph
assert completion['returncodes'] == {'benchmarks': -15, 'native': 0, 'collector': 0}
assert completion['all_complete'] is False
dispatch = get('audits/final_campaign_dispatch.json')
assert dispatch['children']['benchmarks']['pid'] == amend['parents']['3']['pid']
for gpu in ['2', '3']:
    handoff = get(f'audits/balanced_ood_gpu{gpu}_handoff.json')
    state = handoff['state']
    assert handoff['passed'] and state['phase'] == 'available'
    assert state['pid'] == amend['parents'][gpu]['pid']
    assert state['start'] == amend['parents'][gpu]['start']
    assert state['child_exit']['state'] == 'Z' and state['child_exit']['exit_code'] == 0
    assert state['quiet'] >= 2 and state['last_drain_check']['quiet']
    assert not state['last_drain_check']['own_gpu_pids']
    assert status['parents'][gpu]['phase'] == 'available'
replay = get('audits/expanded_supplement_gpu2_protocol_replay.json')
reuse = get('audits/balanced_gpu2_existing_replay_reuse.json')
assert replay['passed'] and replay['actual_gpu'] == '2' and replay['n'] == 1000
assert reuse['passed'] and reuse['actual_replay_sha256'] == sources['audits/expanded_supplement_gpu2_protocol_replay.json']
collection = get('audits/final_collection_status.json')
assert collection['all_complete'] and collection['plan_sha256'] == ph
for path in ['results/confirmation_analysis.json', 'results/confirmation_analysis_strict.json',
             'audits/final_native_comparison.json', 'audits/final_native_comparison_strict.json']:
    assert get(path)['plan_sha256'] == ph
early = get('audits/primary_math_five_seed_early_complete.json')
assert early['plan_sha256'] == ph and early['statistics_source_sha256'] == plan['analysis_script_sha256']
main = get('results/confirmation_analysis.json')
primary = plan['primary_family']
full = main['datasets']['math_reused5000']
remaining = main['validation_excluded_sensitivity']
assert full['families'][primary]['vs_base'] == early['results']['full5000']['effect']
assert remaining['families'][primary]['vs_base'] == early['results']['remaining3500']['effect']
assert full['base_correct'] == early['results']['full5000']['base_correct']
assert remaining['base_correct'] == early['results']['remaining3500']['base_correct']
assert [full['families'][primary]['per_seed'][str(seed)]['correct'] for seed in plan['seeds']] == early['results']['full5000']['per_seed_correct']
assert [remaining['families'][primary]['per_seed_correct'][str(seed)] for seed in plan['seeds']] == early['results']['remaining3500']['per_seed_correct']
for name, digest in early['identities'].items():
    assert sha(S/'results/evaluation/math_reused5000'/name/'math_predictions.jsonl') == digest
for path in ['audits/final_math_overlap_replay.json', 'audits/final_native_baseline_replay.json']:
    assert get(path)['passed']
manifest = get('results/confirmation_manifest.json')
for dataset in plan['datasets']:
    for name in manifest:
        folder = f'results/evaluation/{dataset}/{name}'
        summary = get(folder+'/math_summary.json')
        assert summary['samples'] == amend['dataset_sizes'][dataset]
        for filename in ['math_predictions.jsonl', 'math_manifest.json']:
            path = folder+'/'+filename
            sources[path] = sha(S/path)
assert all(t['status'] == 'complete' and t['returncode'] == 0
           for t in status['tasks'] if t['role'] == 'main')
write(S/'audits/final_campaign_execution_reconciliation.json', dict(
    time=time.time(), passed=True, main_execution_complete=True,
    supplement_gpu_complete=status['supplement_gpu_complete'], plan_sha256=ph,
    original_returncodes=completion['returncodes'], original_all_complete=False,
    early_primary_full5000_and_remaining3500_exact_reproduction=True,
    replacement_status=status['status'], script_sha256=sha(__file__), source_sha256=sources,
    scope='Original CPU benchmark dispatcher was intentionally retired only after its unchanged MATH child exited0 and GPU drained. Replacement completed every frozen main OOD task. Original native and collector exited0. Original nonzero returncodes preserved. Completion is not statistical efficacy or manual scoring approval.'))
print('Main execution reconciliation passed; original returncodes preserved.')
