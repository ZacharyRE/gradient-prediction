"""Five-seed development followup, preserving all seeds and paired endpoint controls."""
import numpy as np
from common import *

plan=json.loads((S/'audits/teacher_average_seed_extension_plan.json').read_text());seeds=plan['fixed_seed_list'];root=S/'results/evaluation/dev'
families={'average1234':'avg_teacher7_1234_s{seed}_epoch4','epoch4':'teacher_all_lr5e5_s{seed}_epoch4'}
missing=[template.format(seed=seed) for template in families.values() for seed in seeds if not (root/template.format(seed=seed)/'math_summary.json').exists()]
if missing:print(json.dumps(dict(pending=missing)));raise SystemExit(0)
replay=json.loads((S/'audits/sampling_restart_protocol_replay.json').read_text());assert replay['passed']
base=read(root/'base/math_predictions.jsonl');b=np.array([r['correct'] for r in base],int)
def summarize(d):
 rng=np.random.default_rng(20261021);qb=[];sb=[]
 for _ in range(10000):
  ix=rng.integers(d.shape[1],size=d.shape[1]);si=rng.integers(d.shape[0],size=d.shape[0]);qb.append(d[:,ix].mean()*100);sb.append(d[si][:,ix].mean()*100)
 return dict(mean_delta_pp=float(d.mean()*100),per_seed_delta_pp=(d.mean(1)*100).tolist(),all_seeds_positive=bool((d.mean(1)>0).all()),question_ci95_pp=np.quantile(qb,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sb,[.025,.975]).tolist())
out={};vectors={}
for family,template in families.items():
 rows=[];matrix=[]
 for seed in seeds:
  name=template.format(seed=seed);p=root/name/'math_predictions.jsonl';rr=read(p)
  assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
  manifest=json.loads((S/f'results/training/teacher_all_lr5e5_s{seed}/manifest.json').read_text());assert manifest['data_sha']==plan['data_sha256']
  for k,v in dict(seed=seed,lr=5e-5,rank=16,modules='qkvo',epochs=8,stop=4,batch=16).items():assert manifest['arguments'][k]==v,(seed,k)
  a=np.array([r['correct'] for r in rr],int);d=a-b;matrix.append(d)
  rows.append(dict(seed=seed,model=name,correct=int(a.sum()),delta_pp=float(d.mean()*100),wins=int((d>0).sum()),losses=int((d<0).sum()),predictions_sha256=sha(p)))
 vectors[family]=np.stack(matrix);out[family]=dict(per_seed=rows,vs_base=summarize(vectors[family]))
write(S/'audits/teacher_average_five_seed_development.json',dict(time=time.time(),n=len(base),base_correct=int(b.sum()),seeds=seeds,families=out,average_minus_epoch4=summarize(vectors['average1234']-vectors['epoch4']),plan_sha256=sha(S/'audits/teacher_average_seed_extension_plan.json'),protocol_replay_sha256=sha(S/'audits/sampling_restart_protocol_replay.json'),scope='Prospective additional-seed development followup of selected candidate, not frozen independent confirmation. Everyseed included, no endpoint/seed switching. FinalOOD/full5000 still governed by separatefreeze.'))
print(json.dumps(out,indent=2))
