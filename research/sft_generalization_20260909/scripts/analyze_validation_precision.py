"""Question-cluster precision and development/validation contrast; no final scores."""
from collections import Counter
import numpy as np
from common import *
datasets=['dev','math_reused1500'];stats={};matrices={};inputs={}
for dataset in datasets:
 src=S/f'data/{dataset}.jsonl';data=read(src);root=S/'results/evaluation'/dataset
 base=read(root/'base/math_predictions.jsonl');b=np.array([r['correct'] for r in base],int)
 inputs[dataset]=dict(data_sha256=sha(src),base_predictions_sha256=sha(root/'base/math_predictions.jsonl'),n=len(data),base_correct=int(b.sum()),subjects=dict(Counter(r['type'] for r in data)),levels=dict(Counter(r['level'] for r in data)))
 for family,template in [('average1234','avg_teacher7_1234_s{seed}_epoch4'),('epoch4','teacher_all_lr5e5_s{seed}_epoch4')]:
  delta=[];hashes={}
  for seed in range(43,48):
   name=template.format(seed=seed);p=root/name/'math_predictions.jsonl';rows=read(p)
   assert [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base]
   delta.append(np.array([r['correct'] for r in rows],int)-b);hashes[name]=sha(p)
  d=np.stack(delta);m=d.mean(0);matrices[dataset,family]=d
  stats[dataset+'_'+family]=dict(mean_delta_pp=float(m.mean()*100),mean_discordance=float(np.abs(d).mean()),question_cluster_sd=float(m.std(ddof=1)),normal_approx_halfwidth95_pp={str(n):float(1.96*m.std(ddof=1)/np.sqrt(n)*100) for n in [500,1500,5000]},model_predictions_sha256=hashes)
qsets=[{''.join(r['problem'].split()) for r in read(S/f'data/{n}.jsonl')} for n in datasets]
assert not(qsets[0]&qsets[1])
contrasts={}
for family in ['average1234','epoch4']:
 d=matrices['dev',family];v=matrices['math_reused1500',family];rng=np.random.default_rng(20261024);qb=[];sb=[]
 for _ in range(10000):
  di=rng.integers(d.shape[1],size=d.shape[1]);vi=rng.integers(v.shape[1],size=v.shape[1]);si=rng.integers(5,size=5)
  qb.append((v[:,vi].mean()-d[:,di].mean())*100);sb.append((v[si][:,vi].mean()-d[si][:,di].mean())*100)
 contrasts[family]=dict(validation_minus_dev_delta_pp=float((v.mean()-d.mean())*100),question_ci95_pp=np.quantile(qb,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sb,[.025,.975]).tolist())
write(S/'audits/validation_precision.json',dict(time=time.time(),script_sha256=sha(__file__),inputs=inputs,exact_whitespace_problem_overlap=0,families=stats,validation_minus_dev=contrasts,scope='Descriptive precision using observed selected-candidate question-cluster variation. Hypothetical sample-size widths assume same iid question distribution and fixed five seeds; not observed5000 outcomes, no prospective power guarantee, no selection correction. Independent question resampling across disjoint datasets with shared seed resampling preserves model pairing. Difference interval crossing zero does not establish distribution equivalence.'))
print(json.dumps(dict(contrasts=contrasts,precision={k:v['normal_approx_halfwidth95_pp'] for k,v in stats.items()}),indent=2))
