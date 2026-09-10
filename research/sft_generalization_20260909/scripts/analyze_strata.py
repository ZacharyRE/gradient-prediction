"""Descriptive error transitions; never use OOD strata for recipe selection."""
from collections import Counter
import numpy as np
from common import *
data=read(S/'data/dev.jsonl');root=S/'results/evaluation/dev';base=read(root/'base/math_predictions.jsonl');out=[]
for p in sorted(root.glob('*/math_summary.json')):
 rr=read(p.with_name('math_predictions.jsonl'));assert len(rr)==len(data) and [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
 for column in ['type','level']:
  for value in sorted({r[column] for r in data}):
   idx=[i for i,r in enumerate(data) if r[column]==value];b=np.array([base[i]['correct'] for i in idx],int);a=np.array([rr[i]['correct'] for i in idx],int);d=a-b
   rng=np.random.default_rng(20260909);boots=d[rng.integers(len(d),size=(3000,len(d)))].mean(1)*100
   out.append(dict(name=p.parent.name,column=column,value=value,n=len(idx),base_correct=int(b.sum()),correct=int(a.sum()),delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boots,[.025,.975]).tolist(),wins=int(((a==1)&(b==0)).sum()),losses=int(((a==0)&(b==1)).sum())))
write(S/'results/dev_strata.json',dict(time=time.time(),status='Descriptive development analysis; unadjusted intervals across many strata are not causal or confirmatory evidence.',rows=out))
for r in out:
 if r['name']=='sample_all_lr1e5_s43_epoch2' and r['column']=='type':print(json.dumps(r))
