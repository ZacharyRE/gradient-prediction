"""Descriptive base-model target compatibility, including same-question pairs."""
import numpy as np
from common import *
scores_path=S/'results/candidate_nll/reused_pool/scores.jsonl'
scores={(r['original_index'],r['target_sha256']):r for r in read(scores_path)}
selected={};summary={}
for name in ['sample_all','teacher_all','teacher32_all']:
 rows=read(S/f'data/{name}.jsonl');lookup={}
 for r in rows:
  key=(r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest())
  assert key in scores;lookup[key[0]]=scores[key]
 selected[name]=lookup;v=list(lookup.values())
 summary[name]=dict(n=len(v),example_mean_nll=float(np.mean([x['nll_mean'] for x in v])),token_mean_nll=sum(x['nll_sum'] for x in v)/sum(x['tokens'] for x in v),mean_target_tokens=float(np.mean([x['tokens'] for x in v])),nll_quantiles=np.quantile([x['nll_mean'] for x in v],[0,.25,.5,.75,.9,1]).tolist(),dataset_sha256=sha(S/f'data/{name}.jsonl'))
pairs=[]
for left,right in [('teacher_all','sample_all'),('teacher32_all','sample_all'),('teacher32_all','teacher_all')]:
 ix=sorted(set(selected[left])&set(selected[right]));rng=np.random.default_rng(20260909)
 d=np.array([selected[left][i]['nll_mean']-selected[right][i]['nll_mean'] for i in ix]);boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
 pairs.append(dict(left=left,right=right,n=len(ix),mean_nll_difference=float(d.mean()),question_ci95=np.quantile(boot,[.025,.975]).tolist(),left_higher=int((d>0).sum()),left_mean_tokens=float(np.mean([selected[left][i]['tokens'] for i in ix])),right_mean_tokens=float(np.mean([selected[right][i]['tokens'] for i in ix]))))
out=dict(time=time.time(),scores_sha256=sha(scores_path),sources=summary,pairs=pairs,interpretation='Teacher-forced completion+EOS NLL under the frozen student. Source comparisons condition on question intersection but differ in text/length/content. Not a causal proof that NLL explains accuracy, nor a process-quality metric. Known bad targets remain in immutable original datasets and are not silently removed.')
write(S/'audits/selected_target_nll.json',out);print(json.dumps(out,indent=2))
