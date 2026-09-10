"""Exploratory transfer strata accounting for baseline sampling coverage."""
import numpy as np
from common import *

def summarize(matrix,mask):
 d=matrix[:,mask];rng=np.random.default_rng(20261015);qb=[];sb=[]
 if not d.shape[1]:return dict(n=0)
 for _ in range(5000):
  ix=rng.integers(d.shape[1],size=d.shape[1]);si=rng.integers(d.shape[0],size=d.shape[0])
  qb.append(d[:,ix].mean()*100);sb.append(d[si][:,ix].mean()*100)
 return dict(n=d.shape[1],mean_delta_pp=float(d.mean()*100),per_seed_delta_pp=(d.mean(1)*100).tolist(),question_ci95_pp=np.quantile(qb,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sb,[.025,.975]).tolist())

train_data=read(S/'data/train_free_probe256.jsonl');dev_data=read(S/'data/dev.jsonl')
sampling=read(S/'results/sampling/student_epoch2_seed_stability/base/predictions.jsonl')
trainbase=read(S/'results/evaluation/train_free_probe256/base/math_predictions.jsonl')
devbase=read(S/'results/evaluation/dev/base/math_predictions.jsonl')
assert len(trainbase)==256 and len(devbase)==500
sample_train_ids={r['original_index'] for r in read(S/'data/sample_all.jsonl')}
assert all(r['original_index'] in sample_train_ids for r in train_data)
ids=np.array([r['index'] for r in sampling]);covered=np.array([r['mean_single_sample_accuracy']>0 for r in sampling])
tb=np.array([r['correct'] for r in trainbase],int);db=np.array([r['correct'] for r in devbase],int)
for r in sampling:assert r['sample_hash']==devbase[r['index']]['sample_hash']
trainmasks=dict(all=np.ones(256,bool),baseline_correct=tb==1,baseline_wrong_with_accepted_student_target=tb==0)
devmasks=dict(all500=np.ones(500,bool),baseline_correct500=db==1,baseline_wrong500=db==0)
for name,sub in [('sampling_subset_all',np.ones(len(ids),bool)),('sampling_subset_greedy_correct',db[ids]==1),('sampling_subset_greedy_wrong_any_correct_of8',(db[ids]==0)&covered),('sampling_subset_greedy_wrong_none_correct_of8',(db[ids]==0)&~covered)]:
 mask=np.zeros(500,bool);mask[ids[sub]]=True;devmasks[name]=mask
families={}
for prefix,epoch in [('sample_all_lr1e5',2),('teacher_all_lr5e5',4)]:
 train=[];dev=[];names=[]
 for seed in [43,44,45]:
  name=f'{prefix}_s{seed}_epoch{epoch}';names.append(name)
  path=S/'results/evaluation/train_free_probe256'/name
  assert (path/'math_summary.json').exists(),f'Wait for all frozen seeds: {name}'
  tr=read(path/'math_predictions.jsonl');dr=read(S/'results/evaluation/dev'/name/'math_predictions.jsonl')
  assert [r['sample_hash'] for r in tr]==[r['sample_hash'] for r in trainbase]
  assert [r['sample_hash'] for r in dr]==[r['sample_hash'] for r in devbase]
  train.append(np.array([r['correct'] for r in tr],int)-tb);dev.append(np.array([r['correct'] for r in dr],int)-db)
 train=np.stack(train);dev=np.stack(dev)
 families[prefix]=dict(models=names,seen_training={n:summarize(train,m) for n,m in trainmasks.items()},heldout_development={n:summarize(dev,m) for n,m in devmasks.items()})
out=dict(time=time.time(),families=families,limitations=[
 'Exploratory strata selected using baseline outputs only; no OOD data. Intervals unadjusted for subgroup comparisons.',
 'Seen-training probe intentionally balances128 historical successes/failures and lies in three-source training intersection. It is not a representative train sample.',
 'Heldout any-correct-of8 conditions on observed baseline sampling coverage, not exact success probability or strict process-valid target availability. It does not match teacher coverage, difficulty, topics or length.',
 'Train-vs-heldout differences describe limited transfer; this conditioning does not isolate a unique causal overfitting mechanism. Training success never substitutes for generalization.'
 ])
write(S/'results/coverage_conditioned_transfer.json',out)
print(json.dumps(out,indent=2))
