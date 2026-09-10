"""CPU-only collection of already prospectively defined fixed-endpoint pilots."""
import subprocess
import numpy as np
from common import *

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
pairs = [
    ('teacher_all_qkvo64_lr1e5_s43_epoch4', 'teacher_all_lr1e5_s43_epoch4',
     'Same teacher targets and LR1e-5: trained QKVO rank64 versus rank16'),
    ('teacher_all_qkvo64_lr5e5_s43_epoch4', 'teacher_all_lr5e5_s43_epoch4',
     'Same teacher targets and LR5e-5: trained QKVO rank64 versus rank16'),
    ('teacher_all_dftkl0_lr5e5_s43_epoch4', 'teacher_all_lr5e5_s43_epoch4',
     'Same teacher targets: DFT versus CE'),
    ('teacher_all_dftkl1_lr5e5_s43_epoch4', 'teacher_all_dftkl0_lr5e5_s43_epoch4',
     'Same DFT teacher training: adding forward KL1'),
    ('teacher_all_dftkl1_lr5e5_s43_epoch4', 'teacher_all_kl1_lr5e5_s43_epoch4',
     'Same teacher training and forward KL1: DFT versus CE'),
    ('sample_multi_lr1e5_s43_epoch1', 'sample_repeat_lr1e5_s43_epoch1',
     'Same6252 training slots and problem order: multiple retained texts versus one repeated text')]
names = list(dict.fromkeys(['base'] + [n for t,c,_ in pairs for n in (t,c)]))
root = S/'results/evaluation/dev'
out = S/'results/teacher_followup_control_analysis.json'
assert not out.exists()
while not all((root/n/'math_summary.json').exists() for n in names):
    if time.time() >= DEADLINE or (S/'STOP_DIAGNOSTICS').exists():
        raise SystemExit('Stopped before all fixed pilots completed')
    time.sleep(20)
auditpath = S/'audits/scoring_teacher_followup_controls.json'
if not auditpath.exists():
    command = [sys.executable, str(S/'scripts/score_sensitivity.py'), '--dataset',
               'dev', '--tag', 'teacher_followup_controls', '--models', *names]
    with (S/'logs/strict_teacher_followup_controls.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=max(1, DEADLINE-time.time()))
audit = json.loads(auditpath.read_text())
bridgepath=S/'audits/evaluator_version_bridge.json';bridge=json.loads(bridgepath.read_text())
assert bridge['passed'] and bridge['current_script_sha256']==sha(S/'scripts/evaluate.py')
assert set(audit['models']) == set(names)
assert audit['dataset_sha256'] == sha(S/'data/dev.jsonl')
base_rows = read(root/'base/math_predictions.jsonl')
base_manifest = json.loads((root/'base/math_manifest.json').read_text())
raw, strict, identities = {}, {}, {}
for name in names:
    path = root/name/'math_predictions.jsonl'
    rows = read(path)
    assert len(rows) == 500
    assert [r['sample_hash'] for r in rows] == [r['sample_hash'] for r in base_rows]
    assert audit['models'][name]['predictions_sha256'] == sha(path)
    manifest = json.loads((root/name/'math_manifest.json').read_text())
    assert manifest['script_sha256'] in bridge['allowed_development_script_sha256']
    for key in ['model','data_sha256','schema','engine',
                'batch_invariant','max_tokens','temperature','chunk_size']:
        assert manifest[key] == base_manifest[key], (name,key)
    identities[name] = dict(manifest_sha256=sha(root/name/'math_manifest.json'),
                            predictions_sha256=sha(path), adapter=manifest['adapter'])
    raw[name] = np.array([r['correct'] for r in rows], int)
    strict[name] = np.array(audit['models'][name]['strict_vector'], int)

def compare(d):
    rng = np.random.default_rng(20261027)
    boots = np.concatenate([d[rng.integers(len(d),size=(200,len(d)))].mean(1)*100
                            for _ in range(50)])
    return dict(delta_pp=float(d.mean()*100), ci95_pp=np.quantile(boots,[.025,.975]).tolist(),
                wins=int((d>0).sum()), losses=int((d<0).sum()))

result = {}
for label, vectors in [('raw',raw),('strict',strict)]:
    b = vectors['base']
    result[label] = dict(base_correct=int(b.sum()),
        models={n:dict(correct=int(v.sum()),vs_base=compare(v-b)) for n,v in vectors.items()},
        pairs=[dict(treatment=t,control=c,hypothesis=h,treatment_correct=int(vectors[t].sum()),
                    control_correct=int(vectors[c].sum()),**compare(vectors[t]-vectors[c]))
               for t,c,h in pairs])
write(out,dict(time=time.time(),script_sha256=sha(__file__),n=500,seed=43,
               evaluator_version_bridge_sha256=sha(bridgepath),
               identities=identities,scoring_audit_sha256=sha(auditpath),results=result,
               scope='Fixed prospective pilot endpoints; single seed43 and unadjusted intervals. '
                     'Rank64 changes parameterization/initial A shapes; multiple targets are distinct '
                     'texts, not certified distinct strategies. No held-out general efficacy claim.'))
print(json.dumps(result,indent=2))
