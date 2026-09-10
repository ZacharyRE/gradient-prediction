"""Bounded supplemental evaluation on GPU1 only after original native work drains."""
import signal
import subprocess
from common import *
from freeze_confirmation import provenance

gpu_guard()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', '')
pp = S/'audits/expanded_unaveraged_supplement_plan.json'
plan = json.loads(pp.read_text()); ph = sha(pp)
integration_path = S/'audits/expanded_supplement_analysis_integration.json'
integration = json.loads(integration_path.read_text())
assert integration['passed'] and integration['analysis_script_sha256'] == sha(S/'scripts/analyze_expanded_supplement.py')
names = list(plan['models'].values())
freeze_path = S/'audits/expanded_unaveraged_supplement_frozen.json'
assert not freeze_path.exists()
while not ((S/'results/confirmation_plan.json').exists() and all(
        (S/'results/training'/n.rsplit('_epoch', 1)[0]/'complete.json').exists() for n in names)):
    if time.time() >= plan['start_cutoff']:
        write(S/'audits/expanded_unaveraged_supplement_status.json',
              dict(status='not_started_readiness_cutoff', all_complete=False, plan_sha256=ph))
        raise SystemExit(0)
    time.sleep(10)
models, artifacts, signatures = {}, {}, []
for seed, name in plan['models'].items():
    path = S/'results/training'/name.rsplit('_epoch', 1)[0]/'epoch4'
    evidence, signature = provenance(path, int(seed))
    assert evidence['manifest']['data_sha'] == plan['training_data_sha256']
    assert evidence['ready']['name'] == name and evidence['ready']['epoch'] == 4
    models[name] = str(path); artifacts[name] = evidence; signatures.append(signature)
assert all(x == signatures[0] for x in signatures)
for dataset in plan['datasets']:
    assert sha(S/f'data/{dataset}.jsonl') == plan['data_sha256'][dataset]
    assert not any((S/'results/evaluation'/dataset/n/'math_predictions.jsonl').exists() for n in names)
assert sha(S/'scripts/evaluate.py') == plan['evaluation_script_sha256']
primary_path = S/'results/confirmation_plan.json'
primary = json.loads(primary_path.read_text())
assert sha(S/'audits/final_selection_rule.json') == plan['primary_selection_rule_sha256']
manifest_path = S/'results/expanded_unaveraged_supplement_manifest.json'
write(manifest_path, models)
write(freeze_path, dict(time=time.time(), plan_sha256=ph, primary_plan_sha256=sha(primary_path),
    manifest_sha256=sha(manifest_path), artifacts=artifacts, recipe_signature=signatures[0],
    pipeline_sha256=sha(__file__), analysis_sha256=sha(S/'scripts/analyze_expanded_supplement.py'),
    analysis_integration_sha256=sha(integration_path),
    primary_campaign_unchanged=True,
    scope='Additional unaveraged five-seed family; original16-model main campaign and criteria unchanged.'))

native = S/'audits/final_native_dispatch.json'
while not (native.exists() and json.loads(native.read_text()).get('all_complete')):
    if time.time() >= plan['start_cutoff']:
        write(S/'audits/expanded_unaveraged_supplement_status.json',
              dict(status='not_started_native_cutoff', all_complete=False, plan_sha256=ph))
        raise SystemExit(0)
    time.sleep(10)
assert (S/'STOP_TRAINER').exists()
checks = []; stable = 0
while time.time() < plan['start_cutoff']:
    owned = []
    for line in subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True).splitlines():
        if not line.strip().isdigit(): continue
        try:
            if (Path('/proc')/line.strip()).stat().st_uid == os.getuid(): owned.append(int(line))
        except FileNotFoundError: pass
    free, total = map(int, subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    good = not owned and free >= .30*total+2048
    stable = stable+1 if good else 0
    checks.append(dict(time=time.time(), own_gpu_pids=owned, free_mib=free, quiet=good))
    if stable >= 2: break
    time.sleep(5)
else:
    write(S/'audits/expanded_unaveraged_supplement_status.json',
          dict(status='not_started_gpu_drain_cutoff', all_complete=False, plan_sha256=ph, checks=checks))
    raise SystemExit(0)
write(S/'audits/expanded_unaveraged_supplement_handoff.json',
      dict(time=time.time(), gpu='1', checks=checks, plan_sha256=ph,
           scope='Original native pipeline finished naturally; no existing process signaled.'))
calls = []
def run(dataset, manifest, label):
    gpu_guard(); assert sha(pp) == ph
    assert sha(S/'scripts/evaluate.py') == plan['evaluation_script_sha256']
    command = [sys.executable, str(S/'scripts/evaluate.py'), '--dataset', dataset,
               '--manifest', str(manifest)]
    with (S/'logs/commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), gpu='1', command=command, role='supplement'))+'\n')
    with (S/f'logs/supplement_{label}.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=max(1, min(DEADLINE-60, plan['finish_cutoff'])-time.time()))
        except subprocess.TimeoutExpired:
            # Only this newly created private session, never the previous native owner.
            assert os.getpgid(proc.pid) == proc.pid
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
            rc = 124
    calls.append(dict(time=time.time(), label=label, command=command, returncode=rc))
    write(S/'audits/expanded_unaveraged_supplement_status.json',
          dict(time=time.time(), status='running' if rc == 0 else 'incomplete_error_or_cutoff',
               all_complete=False, plan_sha256=ph, calls=calls))
    assert rc == 0, (label, rc)
    stable = 0; drain_checks = []
    while time.time() < min(DEADLINE-60, plan['finish_cutoff']):
        owned = []
        for line in subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'], text=True).splitlines():
            if not line.strip().isdigit(): continue
            try:
                if (Path('/proc')/line.strip()).stat().st_uid == os.getuid(): owned.append(int(line))
            except FileNotFoundError: pass
        free, total = map(int, subprocess.check_output(['nvidia-smi', '-i', '1',
            '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(','))
        good = not owned and free >= .30*total+2048
        stable = stable+1 if good else 0
        drain_checks.append(dict(time=time.time(), own_gpu_pids=owned, free_mib=free, quiet=good))
        if stable >= 2: break
        time.sleep(5)
    write(S/f'audits/supplement_drain_{label}.json', dict(plan_sha256=ph, checks=drain_checks))
    assert stable >= 2, 'Supplementary between-task natural drain cutoff'

primary43 = primary['families'][primary['primary_family']]['43']
primary_manifest = json.loads((S/'results/confirmation_manifest.json').read_text())
replay_names = {'supplement_gpu1_replay_base': None,
                'supplement_gpu1_replay_nonzero': primary_manifest[primary43]}
rp = S/'results/expanded_supplement_replay_manifest.json'; write(rp, replay_names)
run('dev', rp, 'gpu1_protocol_replay')
matches = []
for alias, reference in zip(replay_names, ['base', primary43]):
    a = S/'results/evaluation/dev'/alias/'math_predictions.jsonl'
    b = S/'results/evaluation/dev'/reference/'math_predictions.jsonl'
    aa, bb = read(a), read(b); assert len(aa) == len(bb) == 500
    fields = ['index', 'sample_hash', 'prediction', 'correct', 'finish_reason', 'generated_tokens']
    assert all(all(x[k] == y[k] for k in fields) for x, y in zip(aa, bb)), (alias, reference)
    matches.append(dict(alias=alias, reference=reference, alias_sha256=sha(a), reference_sha256=sha(b)))
write(S/'audits/expanded_supplement_gpu1_protocol_replay.json',
      dict(time=time.time(), passed=True, n=1000, matches=matches, plan_sha256=ph))
for dataset in plan['datasets']:
    run(dataset, manifest_path, dataset)
write(S/'audits/expanded_unaveraged_supplement_status.json',
      dict(time=time.time(), status='all_gpu_evaluations_complete', all_complete=True,
           plan_sha256=ph, calls=calls, scope='GPU completion only; scoring/statistics/warning review still required.'))
