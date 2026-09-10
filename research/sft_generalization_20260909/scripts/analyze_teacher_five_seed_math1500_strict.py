"""Preserve the raw result; merge hashed existing strict scores for all five seeds."""
import numpy as np
from common import *
root=S/'results/evaluation/math_reused1500';audits={};provenance={}
for tag in ['math_reused1500_fixed_logged','math_reused1500_averages','teacher_five_seed_math1500_extension']:
 p=S/f'audits/scoring_{tag}.json';a=json.loads(p.read_text());assert a['dataset_sha256']==sha(S/'data/math_reused1500.jsonl')
 provenance[tag]=sha(p)
 for name,item in a['models'].items():
  assert item['predictions_sha256']==sha(root/name/'math_predictions.jsonl')
  if name in audits:assert audits[name]['strict_vector']==item['strict_vector']
  audits[name]=item
raw=json.loads((S/'audits/teacher_average_five_seed_math_reused1500.json').read_text());b=np.array(audits['base']['strict_vector'],int)
def summarize(d):
 rng=np.random.default_rng(20261021);qb=[];sb=[]
 for _ in range(10000):
  ix=rng.integers(d.shape[1],size=d.shape[1]);si=rng.integers(d.shape[0],size=d.shape[0]);qb.append(d[:,ix].mean()*100);sb.append(d[si][:,ix].mean()*100)
 return dict(mean_delta_pp=float(d.mean()*100),per_seed_delta_pp=(d.mean(1)*100).tolist(),all_seeds_positive=bool((d.mean(1)>0).all()),question_ci95_pp=np.quantile(qb,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sb,[.025,.975]).tolist())
out={};vectors={}
for family,f in raw['families'].items():
 rows=[];matrix=[]
 for r in f['per_seed']:
  name=r['model'];assert r['predictions_sha256']==audits[name]['predictions_sha256']
  a=np.array(audits[name]['strict_vector'],int);d=a-b;matrix.append(d)
  rows.append(dict(seed=r['seed'],model=name,correct=int(a.sum()),delta_pp=float(d.mean()*100),wins=int((d>0).sum()),losses=int((d<0).sum())))
 vectors[family]=np.stack(matrix);out[family]=dict(per_seed=rows,vs_base=summarize(vectors[family]))
write(S/'audits/teacher_average_five_seed_math_reused1500_strict.json',dict(time=time.time(),script_sha256=sha(__file__),raw_analysis_sha256=sha(S/'audits/teacher_average_five_seed_math_reused1500.json'),strict_audit_sha256=provenance,n=len(b),base_correct=int(b.sum()),seeds=raw['seeds'],families=out,average_minus_epoch4=summarize(vectors['average1234']-vectors['epoch4']),scope='Strict last-box sensitivity for selected-candidate reused validation, not independent confirmation. Original raw scoring preserved.'))
print(json.dumps(out,indent=2))
