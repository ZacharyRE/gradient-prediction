"""Five-seed paired treatment effects within each inference backend."""
import numpy as np
import argparse
from common import *
parser=argparse.ArgumentParser();parser.add_argument('--strict',action='store_true');args=parser.parse_args()
planpath=S/'results/confirmation_plan.json';plan=json.loads(planpath.read_text());assert sha(__file__)==plan['native_analysis_script_sha256']
native=S/'results/native/final_dev500';vllm=S/'results/evaluation/dev'
mapping=plan['families'][plan['primary_family']];names=['base']+[mapping[str(s)] for s in plan['seeds']]
missing=[n for n in names if not (native/n/'summary.json').exists() or not (vllm/n/'math_summary.json').exists()]
if missing:print(json.dumps(dict(pending=missing)));raise SystemExit(0)
nb=read(native/'base/predictions.jsonl');vb=read(vllm/'base/math_predictions.jsonl');assert len(nb)==len(vb)==500
assert [r['sample_hash'] for r in nb]==[r['sample_hash'] for r in vb]
oldroot=S/'results/native/dev/base';old=read(oldroot/'predictions.jsonl');assert len(old)==128
nm=json.loads((native/'base/manifest.json').read_text());om=json.loads((oldroot/'manifest.json').read_text())
for k in ['model','adapter','batch','attention','base_dtype','adapter_dtype','compute_autocast','use_cache','temperature','max_new_tokens','data_sha256']:assert nm[k]==om[k],k
fields=['sample_hash','prediction','correct','finish_reason','generated_tokens'];replay={k:sum(a[k]==b[k] for a,b in zip(nb[:128],old)) for k in fields}
write(S/'audits/final_native_baseline_replay.json',dict(time=time.time(),passed=all(v==128 for v in replay.values()),n=128,equal_counts=replay,old_predictions_sha256=sha(oldroot/'predictions.jsonl'),new_predictions_sha256=sha(native/'base/predictions.jsonl')))
assert all(v==128 for v in replay.values()),'Native baseline replay differs'
nbv=np.array([r['correct'] for r in nb],int);vbv=np.array([r['correct'] for r in vb],int);rows=[];nd=[];vd=[]
scoring={}
if args.strict:
 for backend,tag,folder,filename in [('native','final_native_dev500',native,'predictions.jsonl'),('vllm','final_primary_dev500',vllm,'math_predictions.jsonl')]:
  audit=json.loads((S/f'audits/scoring_{tag}.json').read_text());assert audit['script_sha256']==plan['strict_scoring_script_sha256']
  assert audit['dataset_sha256']==sha(S/'data/dev.jsonl') and set(audit['models'])==set(names)
  assert all(audit['models'][name]['predictions_sha256']==sha(folder/name/filename) for name in names)
  scoring[backend]=audit['models']
 nbv=np.array(scoring['native']['base']['strict_vector'],int);vbv=np.array(scoring['vllm']['base']['strict_vector'],int)
for seed in plan['seeds']:
 name=mapping[str(seed)];nr=read(native/name/'predictions.jsonl');vr=read(vllm/name/'math_predictions.jsonl');assert len(nr)==len(vr)==500
 assert [r['sample_hash'] for r in nr]==[r['sample_hash'] for r in vr]==[r['sample_hash'] for r in nb]
 nm=json.loads((native/name/'manifest.json').read_text());assert nm['adapter_loaded_exactly'] and nm['adapter']['files']==plan['adapter_files_sha256'][name]
 vm=json.loads((vllm/name/'math_manifest.json').read_text());assert vm['adapter']['files']==plan['adapter_files_sha256'][name]
 assert vm['data_sha256']==sha(S/'data/dev.jsonl')
 bridgepath=S/'audits/evaluator_version_bridge.json';assert sha(bridgepath)==plan['evaluator_version_bridge_sha256']
 assert vm['script_sha256'] in plan['allowed_development_script_sha256']
 for key,value in plan['evaluation_protocol'].items():
  if key!='script_sha256':assert vm[key]==value,(name,key)
 assert vm['model']['files']==plan['base_model_files_sha256']
 nv=np.array(scoring['native'][name]['strict_vector'] if args.strict else [r['correct'] for r in nr],int);vv=np.array(scoring['vllm'][name]['strict_vector'] if args.strict else [r['correct'] for r in vr],int);nd.append(nv-nbv);vd.append(vv-vbv)
 rows.append(dict(seed=seed,name=name,native_correct=int(nv.sum()),vllm_correct=int(vv.sum()),native_delta_pp=float((nv-nbv).mean()*100),vllm_delta_pp=float((vv-vbv).mean()*100),native_predictions_sha256=sha(native/name/'predictions.jsonl'),vllm_predictions_sha256=sha(vllm/name/'math_predictions.jsonl'),exact_backend_texts=sum(a['prediction']==b['prediction'] for a,b in zip(nr,vr))))
def summarize(d):
 rng=np.random.default_rng(20261025);qb=[];sb=[]
 for _ in range(10000):
  ix=rng.integers(d.shape[1],size=d.shape[1]);si=rng.integers(d.shape[0],size=d.shape[0]);qb.append(d[:,ix].mean()*100);sb.append(d[si][:,ix].mean()*100)
 return dict(mean_delta_pp=float(d.mean()*100),per_seed_delta_pp=(d.mean(1)*100).tolist(),all_seeds_positive=bool((d.mean(1)>0).all()),question_ci95_pp=np.quantile(qb,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sb,[.025,.975]).tolist())
nd=np.stack(nd);vd=np.stack(vd)
output='final_native_comparison_strict.json' if args.strict else 'final_native_comparison.json'
write(S/'audits'/output,dict(time=time.time(),scoring='strict_last_box' if args.strict else 'original',script_sha256=sha(__file__),plan_sha256=sha(planpath),n=500,native_base_correct=int(nbv.sum()),vllm_base_correct=int(vbv.sum()),seeds=plan['seeds'],rows=rows,native_vs_native_base=summarize(nd),vllm_vs_vllm_base=summarize(vd),native_minus_vllm_treatment_effect=summarize(nd-vd),scope='Reused-development backend sensitivity of all five frozen primary models. Backend, batch and arithmetic differ jointly; compare effects within each backend. A difference interval crossing zero does not establish backend equivalence. This is not an additional fresh efficacy test.'))
print(json.dumps(dict(rows=rows,native=summarize(nd)),indent=2))
