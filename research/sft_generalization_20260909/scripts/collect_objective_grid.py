"""Fixed 2x3 objective/coefficient development grid; CPU-only, all cells retained."""
import subprocess
import numpy as np
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
grid={('ce','0'):'teacher_all_lr5e5_s43_epoch4'}
for coefficient in ['01','1']:
    grid['ce',coefficient]=f'teacher_all_kl{coefficient}_lr5e5_s43_epoch4'
for coefficient in ['0','01','1']:
    grid['dft',coefficient]=f'teacher_all_dftkl{coefficient}_lr5e5_s43_epoch4'
names=['base',*grid.values()]
root=S/'results/evaluation/dev'
output=S/'results/objective_coefficient_grid.json'
assert not output.exists()
while not all((root/name/'math_summary.json').exists() for name in names):
    if time.time()>=DEADLINE or (S/'STOP_DIAGNOSTICS').exists():
        raise SystemExit('Stopped before grid completion')
    time.sleep(20)
auditpath=S/'audits/scoring_objective_coefficient_grid.json'
if not auditpath.exists():
    command=[sys.executable,str(S/'scripts/score_sensitivity.py'),'--dataset','dev',
             '--tag','objective_coefficient_grid','--models',*names]
    with (S/'logs/strict_objective_coefficient_grid.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                       timeout=max(1,DEADLINE-time.time()))
audit=json.loads(auditpath.read_text())
bridgepath=S/'audits/evaluator_version_bridge.json';bridge=json.loads(bridgepath.read_text())
assert bridge['passed'] and bridge['current_script_sha256']==sha(S/'scripts/evaluate.py')
assert set(audit['models'])==set(names) and audit['dataset_sha256']==sha(S/'data/dev.jsonl')
reference=json.loads((root/'base/math_manifest.json').read_text())
base_rows=read(root/'base/math_predictions.jsonl')
raw,strict,identities={},{},{}
for name in names:
    path=root/name/'math_predictions.jsonl';rows=read(path)
    assert len(rows)==500 and [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base_rows]
    assert audit['models'][name]['predictions_sha256']==sha(path)
    manifest=json.loads((root/name/'math_manifest.json').read_text())
    assert manifest['script_sha256'] in bridge['allowed_development_script_sha256']
    for key in ['model','data_sha256','schema','engine','batch_invariant',
                'max_tokens','temperature','chunk_size']:
        assert manifest[key]==reference[key],(name,key)
    if name!='base':
        training=S/'results/training'/name.rsplit('_epoch',1)[0]
        if '_kl' in name or 'dftkl01' in name or 'dftkl1_' in name:
            anchoring=json.loads((training/'anchoring_audit.json').read_text())
            assert anchoring['reference_unchanged'] and anchoring['first_batch']['initial_logits_max_abs']==0
    raw[name]=np.array([r['correct'] for r in rows],int)
    strict[name]=np.array(audit['models'][name]['strict_vector'],int)
    identities[name]=dict(predictions_sha256=sha(path),manifest_sha256=sha(root/name/'math_manifest.json'))

def compare(d):
    rng=np.random.default_rng(20261028)
    boots=np.concatenate([d[rng.integers(len(d),size=(200,len(d)))].mean(1)*100 for _ in range(50)])
    return dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boots,[.025,.975]).tolist())

results={}
for scoring,v in [('raw',raw),('strict',strict)]:
    contrasts=[]
    for coefficient in ['0','01','1']:
        t,c=grid['dft',coefficient],grid['ce',coefficient]
        contrasts.append(dict(kind='DFT_minus_CE',coefficient=coefficient,treatment=t,control=c,**compare(v[t]-v[c])))
    for coefficient in ['01','1']:
        for objective in ['ce','dft']:
            t,c=grid[objective,coefficient],grid[objective,'0']
            contrasts.append(dict(kind='KL_minus_noKL',objective=objective,coefficient=coefficient,
                                  treatment=t,control=c,**compare(v[t]-v[c])))
        d=(v[grid['dft',coefficient]]-v[grid['dft','0']])-(v[grid['ce',coefficient]]-v[grid['ce','0']])
        contrasts.append(dict(kind='objective_by_KL_interaction',coefficient=coefficient,
                              formula='(DFT+KL - DFT) - (CE+KL - CE)',**compare(d)))
    results[scoring]=dict(base_correct=int(v['base'].sum()),
        models={n:dict(correct=int(x.sum()),vs_base=compare(x-v['base'])) for n,x in v.items()},
        contrasts=contrasts)
write(output,dict(time=time.time(),script_sha256=sha(__file__),scoring_audit_sha256=sha(auditpath),
    evaluator_version_bridge_sha256=sha(bridgepath),
    prospective_plans={name:sha(S/'audits'/name) for name in ['teacher_followup_controls_plan.json','dft_kl01_control_plan.json']},
    seed=43,n=500,identities=identities,results=results,
    scope='Selected development data, one seed, fixed epoch4 and unadjusted paired intervals. '
          'No OOD or general efficacy claim. Coefficient0.1 was declared before any DFT training/scores; '
          'CE outcomes were already known. Does not reproduce paper-scale models/data/batch/decoding.'))
print(json.dumps(results,indent=2))
