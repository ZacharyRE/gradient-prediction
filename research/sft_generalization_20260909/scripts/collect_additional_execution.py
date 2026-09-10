"""CPU execution amendment: advance existing strict scoring and case preparation."""
import subprocess
from common import *

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ap = S/'audits/additional_collection_execution_amendment.json'
amend = json.loads(ap.read_text()); ah = sha(ap)
assert amend['execution_script_sha256'] == sha(__file__)
assert sha(S/'results/confirmation_plan.json') == amend['primary_plan_sha256']
sp = S/'audits/expanded_unaveraged_supplement_plan.json'
assert sha(sp) == amend['supplement_plan_sha256']
supp_plan = json.loads(sp.read_text())
hashes = amend['unchanged_dependency_sha256']
for name, digest in hashes.items():
    assert sha(S/'scripts'/name) == digest
audit = S/'audits/additional_diagnostic_collection.json'
old = json.loads(audit.read_text())
assert not old['done'] and not old['calls']
assert old['script_sha256'] == amend['original_collector_sha256']
done = []; calls = []; pre_scored = []

def get(path):
    p = S/path
    return json.loads(p.read_text()) if p.exists() else {}

def publish(status, **extra):
    write(audit, dict(time=time.time(), status=status,
        all_complete=status == 'prepared_for_manual_review', done=done, calls=calls,
        pre_scored_supplement_datasets=pre_scored, script_sha256=sha(__file__),
        dependency_script_sha256=hashes, execution_amendment_sha256=ah,
        scope='CPU execution timing only. Frozen scorers, analyses, models and criteria unchanged. No automatic review approval.', **extra))

def run(name, args=None, label=None):
    label = label or name
    if label in done:
        return
    assert sha(ap) == ah and sha(S/'scripts'/name) == hashes[name]
    command = [sys.executable, str(S/'scripts'/name), *(args or [])]
    with (S/f'logs/additional_execution_{Path(label).stem}.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=max(1, DEADLINE-60-time.time()))
    calls.append(dict(time=time.time(), script=name, label=label,
                      command=command, returncode=result.returncode))
    if result.returncode:
        publish('error', failed_script=name)
        raise SystemExit(result.returncode)
    done.append(label)
    publish('waiting_or_collecting')

def score_ready_supplement():
    names = list(supp_plan['models'].values())
    for dataset in supp_plan['datasets']:
        if dataset in pre_scored:
            continue
        root = S/'results/evaluation'/dataset
        if not all((root/n/'math_summary.json').exists() for n in names):
            continue
        assert get('audits/expanded_supplement_gpu2_protocol_replay.json').get('passed')
        assert sha(S/f'data/{dataset}.jsonl') == supp_plan['data_sha256'][dataset]
        path = S/f'audits/scoring_supplement_{dataset}.json'
        assert not path.exists(), 'Single controller owns supplementary strict scoring'
        run('score_sensitivity.py', ['--dataset', dataset, '--tag', 'supplement_'+dataset,
                                   '--models', *names], 'prestrict_'+dataset)
        record = json.loads(path.read_text())
        assert record['script_sha256'] == supp_plan['strict_script_sha256']
        assert record['dataset_sha256'] == supp_plan['data_sha256'][dataset]
        assert set(record['models']) == set(names)
        for name in names:
            assert record['models'][name]['predictions_sha256'] == sha(root/name/'math_predictions.jsonl')
        pre_scored.append(dataset)
        publish('waiting_or_collecting')

publish('waiting_or_collecting')
try:
    while time.time() < DEADLINE-60:
        main = get('audits/final_collection_status.json').get('all_complete', False)
        supp = get('audits/expanded_unaveraged_supplement_status.json')
        native = get('audits/native_target_control_status.json')
        supp_terminal = supp.get('all_complete') or supp.get('status', '').startswith('not_started') or supp.get('status') == 'incomplete_error_or_cutoff'
        native_terminal = native.get('all_complete') or native.get('status', '').startswith('not_started') or native.get('status') == 'incomplete_error_or_cutoff'
        if native.get('all_complete'):
            run('analyze_native_target_controls.py')
        score_ready_supplement()
        benchmark_ready = all(get(p).get('plan_sha256') == amend['primary_plan_sha256']
                              for p in ['results/confirmation_analysis.json', 'results/confirmation_analysis_strict.json'])
        if benchmark_ready:
            run('prepare_final_transition_review.py')
        if main:
            run('prepare_final_warning_review.py')
        if main and supp_terminal:
            if supp.get('all_complete'):
                assert set(pre_scored) == set(supp_plan['datasets'])
                run('analyze_expanded_supplement.py')
                run('prepare_supplement_warning_review.py')
                run('analyze_supplement_math_sensitivity.py')
            run('analyze_final_math_strata.py')
            run('analyze_math_validation_transfer.py')
            run('prepare_minerva_unit_review.py')
        if main and supp_terminal and native_terminal:
            publish('prepared_for_manual_review', supplement_complete=bool(supp.get('all_complete')),
                    native_control_complete=bool(native.get('all_complete')))
            break
        publish('waiting_or_collecting')
        time.sleep(10)
    else:
        publish('global_deadline_incomplete')
except BaseException as exc:
    publish('error', error=repr(exc))
    raise
