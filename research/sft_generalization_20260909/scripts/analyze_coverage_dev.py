"""Raw and strict sensitivity for all six prospective 7B scale pilot endpoints."""
import numpy as np
from common import *
auditpath=S/'audits/scoring_coverage_dev_fixed.json'
while not auditpath.exists() and time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():time.sleep(5)
assert auditpath.exists()
audit=json.loads(auditpath.read_text());root=S/'results/evaluation/dev'
bridgepath=S/'audits/evaluator_version_bridge.json';bridge=json.loads(bridgepath.read_text())
assert bridge['passed'] and bridge['current_script_sha256']==sha(S/'scripts/evaluate.py')
base_rows=read(root/'base/math_predictions.jsonl');base_manifest=json.loads((root/'base/math_manifest.json').read_text())
raw,strict={},{}
assert audit['dataset_sha256']==sha(S/'data/dev.jsonl')
for name,record in audit['models'].items():
    path=root/name/'math_predictions.jsonl';rows=read(path)
    assert len(rows)==500 and [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base_rows]
    assert record['predictions_sha256']==sha(path)
    manifest=json.loads((root/name/'math_manifest.json').read_text())
    assert manifest['script_sha256'] in bridge['allowed_development_script_sha256']
    for key in ['model','data_sha256','engine','batch_invariant','max_tokens','temperature','chunk_size']:
        assert manifest[key]==base_manifest[key]
    raw[name]=np.array([r['correct'] for r in rows],int)
    strict[name]=np.array(record['strict_vector'],int)
def avg(source):return 'avg_teacher7_expansion_'+source+'_1234_s43_epoch4'
def endpoint(source):return 'teacher7_expansion_'+source+'_lr5e5_s43_epoch4'
pairs=[]
for naming in [avg,endpoint]:
    pairs += [(naming('combined'),naming('repeat_control'),'Expanded coverage versus repetition with same4145slots'),
              (naming('combined'),naming('rawnew_control'),'Generated versus raw targets on identical new questions')]
for source in ['combined','repeat_control','rawnew_control']:
    pairs.append((avg(source),endpoint(source),'Fixed epoch1..4 effective-update average versus fixedepoch4'))
def compare(d):
    rng=np.random.default_rng(20261029)
    boots=np.concatenate([d[rng.integers(len(d),size=(200,len(d)))].mean(1)*100 for _ in range(50)])
    return dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boots,[.025,.975]).tolist(),wins=int((d>0).sum()),losses=int((d<0).sum()))
results={}
for label,v in [('raw',raw),('strict',strict)]:
    results[label]=dict(base_correct=int(v['base'].sum()),
        models={n:dict(correct=int(x.sum()),vs_base=compare(x-v['base'])) for n,x in v.items()},
        pairs=[dict(treatment=t,control=c,hypothesis=h,**compare(v[t]-v[c])) for t,c,h in pairs])
write(S/'results/coverage_dev_fixed_analysis.json',dict(time=time.time(),script_sha256=sha(__file__),
    evaluator_version_bridge_sha256=sha(bridgepath),
    scoring_audit_sha256=sha(auditpath),seed=43,n=500,results=results,
    scope='Single-seed selected-development pilot sensitivity, fixed prospective endpoints, unadjusted intervals. '
          'Global raw Holm sensitivity is separate. Not multi-seed general efficacy evidence.'))
print(json.dumps(results,indent=2))
