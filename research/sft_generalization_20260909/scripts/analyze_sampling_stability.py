"""Fixed epoch2 expected single-draw stability; no best-seed selection."""
import numpy as np
from common import *
def compare(rows,base):
 assert [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base]
 return np.array([r['mean_single_sample_accuracy']-b['mean_single_sample_accuracy'] for r,b in zip(rows,base)])
pilot=S/'results/sampling/student_epoch2_pilot';base=read(pilot/'base/predictions.jsonl');model=read(pilot/'sample_all_lr1e5_s43_epoch2/predictions.jsonl');greedy=read(S/'results/evaluation/dev/base/math_predictions.jsonl');d=compare(model,base);groups={}
for right in [False,True]:
 mask=np.array([greedy[r['index']]['correct']==right for r in base]);dd=d[mask];rng=np.random.default_rng(20260926);boots=dd[rng.integers(len(dd),size=(10000,len(dd)))].mean(1)*100
 groups[str(right)]=dict(n=int(mask.sum()),delta_pp=float(dd.mean()*100),question_ci95_pp=np.quantile(boots,[.025,.975]).tolist())
write(S/'audits/sampling_pilot_baseline_strata.json',dict(time=time.time(),strata_by_baseline_greedy_correct=groups,limitation='Exploratory subgroups on pilot development; intervals unadjusted. Correctness averages finite8 draws; not exact model probability.'))
root=S/'results/sampling/student_epoch2_seed_stability';result=dict(time=time.time(),fixed_seeds=[43,44,45],fixed_epoch=2,phase='Development replication, not OOD confirmation',status='pending')
if (root/'complete.json').exists():
 bb=read(root/'base/predictions.jsonl');assert [r['sample_hash'] for r in bb]==[r['sample_hash'] for r in base];result['baseline_exact_pilot_replay']=bb==base;deltas=[];perseed=[]
 ids=[r['index'] for r in bb];greedy_base=np.array([greedy[i]['correct'] for i in ids],float)
 result['same_subset_baseline_greedy_accuracy']=float(greedy_base.mean())
 for seed in [43,44,45]:
  rr=read(root/f'sample_all_lr1e5_s{seed}_epoch2/predictions.jsonl');dd=compare(rr,bb);deltas.append(dd);rng=np.random.default_rng(20260927);boots=dd[rng.integers(len(dd),size=(10000,len(dd)))].mean(1)*100
  if seed==43:result['seed43_exact_pilot_replay']=rr==model
  gr=read(S/f'results/evaluation/dev/sample_all_lr1e5_s{seed}_epoch2/math_predictions.jsonl');gd=np.array([gr[i]['correct'] for i in ids],float)-greedy_base;interaction=dd-gd;iboot=interaction[rng.integers(len(dd),size=(10000,len(dd)))].mean(1)*100;accuracy=float(np.mean([r['mean_single_sample_accuracy'] for r in rr]))
  perseed.append(dict(seed=seed,delta_pp=float(dd.mean()*100),question_ci95_pp=np.quantile(boots,[.025,.975]).tolist(),accuracy=accuracy,same_subset_greedy_delta_pp=float(gd.mean()*100),sampling_vs_greedy_treatment_interaction_pp=float(interaction.mean()*100),interaction_question_ci95_pp=np.quantile(iboot,[.025,.975]).tolist(),sampling_accuracy_minus_baseline_greedy_pp=float((accuracy-greedy_base.mean())*100)))
 matrix=np.array(deltas);rng=np.random.default_rng(20260928);qboots=[];sqboots=[]
 for _ in range(10000):
  ix=rng.integers(matrix.shape[1],size=matrix.shape[1]);ss=rng.integers(3,size=3);qboots.append(matrix[:,ix].mean()*100);sqboots.append(matrix[ss][:,ix].mean()*100)
 result.update(status='complete',per_seed=perseed,all_seeds_positive=all(x['delta_pp']>0 for x in perseed),mean_delta_pp=float(matrix.mean()*100),question_ci95_pp=np.quantile(qboots,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sqboots,[.025,.975]).tolist(),limitation='Only3 training seeds; seed bootstrap has limited support. Development results remain adaptively chosen; no general efficacy without reserved OOD.')
write(S/'audits/sampling_seed_stability.json',result);print(json.dumps(dict(pilot_strata=groups,stability=result),indent=2))
