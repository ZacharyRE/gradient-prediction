"""Select fixed qualitative cases of three-seed-consistent heldout transitions."""
import random
import numpy as np
from common import *

root=S/'results/evaluation/math_reused1500';data=read(S/'data/math_reused1500.jsonl');base=read(root/'base/math_predictions.jsonl')
auditpath=S/'audits/scoring_math_reused1500_fixed_logged.json';audit=json.loads(auditpath.read_text())
assert audit['dataset_sha256']==sha(S/'data/math_reused1500.jsonl')
for name,record in audit['models'].items():assert record['predictions_sha256']==sha(root/name/'math_predictions.jsonl')
b=np.array(audit['models']['base']['strict_vector'],bool);rng=random.Random(20261019);selected=[];counts={};used=set()
for prefix,epoch in [('sample_all_lr1e5',2),('teacher_all_lr5e5',4)]:
 names=[f'{prefix}_s{seed}_epoch{epoch}' for seed in [43,44,45]];preds={name:read(root/name/'math_predictions.jsonl') for name in names}
 for rr in preds.values():assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
 matrix=np.stack([audit['models'][n]['strict_vector'] for n in names]).astype(bool)
 normal=np.array([base[i]['finish_reason']=='stop' and all(preds[n][i]['finish_reason']=='stop' for n in names) for i in range(len(base))])
 for transition,mask in [('consistent_gain',~b&matrix.all(0)),('consistent_loss',b&~matrix.any(0))]:
  eligible=[int(i) for i in np.flatnonzero(mask&normal) if int(i) not in used];chosen=rng.sample(eligible,min(2,len(eligible)));used.update(chosen)
  group=prefix+'_'+transition;counts[group]=dict(all_consistent=int(mask.sum()),all_four_normal_stop=int((mask&normal).sum()),eligible_after_previous_cases=len(eligible),selected=chosen)
  for i in chosen:
   selected.append(dict(group=group,index=i,sample_hash=base[i]['sample_hash'],problem=data[i]['problem'],reference=data[i]['solution'],type=data[i]['type'],level=data[i]['level'],baseline=base[i],adapted={n:preds[n][i] for n in names},strict_baseline=bool(b[i]),strict_adapted=matrix[:,i].tolist()))
destination=S/'audits/consistent_transition_review.jsonl'
if destination.exists():assert read(destination)==selected
else:jsonl(destination,selected)
write(S/'audits/consistent_transition_review_plan.json',dict(time=time.time(),seed=20261019,n=len(selected),counts=counts,data_sha256=sha(S/'data/math_reused1500.jsonl'),strict_audit_sha256=sha(auditpath),review_sha256=sha(destination),selection='Two fixed random cases per family and consistent-transition direction, requiring all four outputs normal-stop and avoiding already-selected questions. Uses strictlastbox across all three seeds.',limitations='Deliberately selected qualitative transitions, not error prevalence or causal effect estimates. Historically reused1500 only; noOOD. A correct final answer does not automatically certify a baseline or SFT derivation.'))
print(json.dumps(counts,indent=2))
