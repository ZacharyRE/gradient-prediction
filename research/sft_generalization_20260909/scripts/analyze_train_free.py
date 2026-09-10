"""Separate teacher-forced fitting, seen-question generation, and held-out generation."""
import numpy as np
from common import *

def interval(delta,seed=20261011):
 rng=np.random.default_rng(seed);v=np.asarray(delta,dtype=float)
 draws=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)*100
 return np.quantile(draws,[.025,.975]).tolist()

def main():
 root=S/'results/evaluation/train_free_probe256';plan=json.loads((S/'audits/train_free_probe_plan.json').read_text());data=read(S/'data/train_free_probe256.jsonl')
 if not (root/'base/math_summary.json').exists():print('Pending baseline');return
 base=read(root/'base/math_predictions.jsonl');assert len(base)==len(data)==256
 b=np.array([r['correct'] for r in base],int);old=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl')
 replay={k:sum(r[k]==old[d['original_index']][k] for r,d in zip(base,data)) for k in ['prediction','correct']}
 strata={'all':np.ones(256,dtype=bool),'historical_wrong':np.array([not d['historical_base_correct'] for d in data]),'historical_correct':np.array([d['historical_base_correct'] for d in data])};comparisons={};arrays={}
 for name in plan['models']:
  if name=='base' or not (root/name/'math_summary.json').exists():continue
  rr=read(root/name/'math_predictions.jsonl');assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
  x=np.array([r['correct'] for r in rr],int);arrays[name]=x;out={}
  for group,mask in strata.items():
   d=(x-b)[mask];out[group]=dict(n=int(mask.sum()),base_correct=int(b[mask].sum()),sft_correct=int(x[mask].sum()),delta_pp=float(d.mean()*100),question_ci95_pp=interval(d),wins=int(((x==1)&(b==0)&mask).sum()),losses=int(((x==0)&(b==1)&mask).sum()))
  comparisons[name]=out
 families={}
 for prefix,epoch in [('sample_all_lr1e5',2),('teacher_all_lr5e5',4)]:
  names=[f'{prefix}_s{seed}_epoch{epoch}' for seed in [43,44,45]]
  if not all(n in arrays for n in names):continue
  matrix=np.stack([arrays[n]-b for n in names]);rng=np.random.default_rng(20261011);boots=[]
  for _ in range(5000):boots.append(matrix[rng.integers(3,size=3)][:,rng.integers(256,size=256)].mean()*100)
  families[prefix]=dict(models=names,mean_delta_pp=float(matrix.mean()*100),all_seeds_positive=bool((matrix.mean(1)>0).all()),question_ci95_pp=interval(matrix.mean(0)),seed_question_ci95_pp=np.quantile(boots,[.025,.975]).tolist())
 strict={};audit_path=S/'audits/scoring_train_free_probe256.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text());assert audit['dataset_sha256']==sha(S/'data/train_free_probe256.jsonl')
  for name,record in audit['models'].items():assert record['predictions_sha256']==sha(root/name/'math_predictions.jsonl')
  sb=np.array(audit['models']['base']['strict_vector'],int)
  strict=dict(audit_sha256=sha(audit_path),base_correct=int(sb.sum()),per_model={},families={})
  for name in arrays:
   x=np.array(audit['models'][name]['strict_vector'],int);d=x-sb
   strict['per_model'][name]=dict(correct=int(x.sum()),delta_pp=float(d.mean()*100),question_ci95_pp=interval(d))
  for family,record in families.items():
   matrix=np.stack([np.array(audit['models'][n]['strict_vector'],int)-sb for n in record['models']]);rng=np.random.default_rng(20261011);boots=[]
   for _ in range(5000):boots.append(matrix[rng.integers(3,size=3)][:,rng.integers(256,size=256)].mean()*100)
   strict['families'][family]=dict(mean_delta_pp=float(matrix.mean()*100),all_seeds_positive=bool((matrix.mean(1)>0).all()),question_ci95_pp=interval(matrix.mean(0)),seed_question_ci95_pp=np.quantile(boots,[.025,.975]).tolist())
 write(S/'results/train_free_generation_analysis.json',dict(time=time.time(),n=256,historical_baseline_replay_equal_counts=replay,comparisons=comparisons,families=families,strict_sensitivity=strict,missing=[n for n in plan['models'] if n!='base' and n not in arrays],interpretation='Seen-training diagnostic on a deliberately balanced historical-correctness sample; never an independent generalization score. Free-generation gains, if present, must be contrasted with fixed-seed dev/OOD results rather than teacher-forced loss alone.'))
 print(json.dumps(dict(replay=replay,overall={n:r['all'] for n,r in comparisons.items()},families=families),indent=2))

if __name__=='__main__':main()
