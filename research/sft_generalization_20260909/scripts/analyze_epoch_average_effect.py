"""Direct paired averaging effects; significance versus baseline is not a contrast."""
import numpy as np
from common import *

out={}
for dataset in ['dev','math_reused1500']:
 root=S/'results/evaluation'/dataset;out[dataset]={}
 for label,averaged,endpoint in [('student','avg_sample12_s{seed}_epoch2','sample_all_lr1e5_s{seed}_epoch2'),('teacher7','avg_teacher7_1234_s{seed}_epoch4','teacher_all_lr5e5_s{seed}_epoch4')]:
  matrix=[];records=[]
  for seed in [43,44,45]:
   an=averaged.format(seed=seed);en=endpoint.format(seed=seed);ap=root/an/'math_predictions.jsonl';ep=root/en/'math_predictions.jsonl'
   assert ap.with_name('math_summary.json').exists() and ep.with_name('math_summary.json').exists()
   aa=read(ap);ee=read(ep);assert [r['sample_hash'] for r in aa]==[r['sample_hash'] for r in ee]
   d=np.array([int(a['correct'])-int(e['correct']) for a,e in zip(aa,ee)]);matrix.append(d)
   records.append(dict(seed=seed,averaged=an,endpoint=en,delta_pp=float(d.mean()*100),wins=int((d>0).sum()),losses=int((d<0).sum()),averaged_prediction_sha256=sha(ap),endpoint_prediction_sha256=sha(ep)))
  d=np.stack(matrix);rng=np.random.default_rng(20261020);q=[];sq=[]
  for _ in range(10000):
   ix=rng.integers(d.shape[1],size=d.shape[1]);si=rng.integers(3,size=3);q.append(d[:,ix].mean()*100);sq.append(d[si][:,ix].mean()*100)
  out[dataset][label]=dict(n=d.shape[1],records=records,mean_delta_pp=float(d.mean()*100),all_seeds_positive=bool((d.mean(1)>0).all()),question_ci95_pp=np.quantile(q,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sq,[.025,.975]).tolist())
write(S/'audits/epoch_average_paired_effects.json',dict(time=time.time(),results=out,scope='Matched within-training-seed averaged-update vs fixed finalcheckpoint, on development and historically reused1500. Exploratory unadjusted intervals. A positive baseline contrast and a nonsignificant alternative baseline contrast do not themselves prove a positive difference between models.',limits='Exact averaging changes the effective update and inference factor rank; this comparison does not isolate pure variance reduction from every other property of the derived update.'))
print(json.dumps(out,indent=2))
