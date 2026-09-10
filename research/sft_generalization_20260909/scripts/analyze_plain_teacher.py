"""Paired NLL effect of the frozen cosmetic-markup transformation."""
import numpy as np
from common import *
root=S/'results/candidate_nll/teacher32_plain_markup';assert (root/'complete.json').exists()
plan=json.loads((S/'audits/plain_teacher_plan.json').read_text());assert sha(S/'data/teacher32_plain_markup.jsonl')==plan['data_sha256']
assert json.loads((root/'manifest.json').read_text())['input_sha256']==plan['data_sha256']
oldbank={(r['original_index'],r['target_sha256']):r for r in read(S/'results/candidate_nll/reused_pool/scores.jsonl')}
oldrows=read(S/'data/teacher32_all.jsonl');newrows=read(S/'data/teacher32_plain_markup.jsonl');new=read(root/'scores.jsonl');assert len(oldrows)==len(newrows)==len(new)==1637
old=[]
for a,b,r in zip(oldrows,newrows,new):
 assert a['original_index']==b['original_index']==r['original_index'] and a['problem']==b['problem']
 assert hashlib.sha256(b['solution'].encode()).hexdigest()==r['target_sha256']
 old.append(oldbank[(a['original_index'],hashlib.sha256(a['solution'].encode()).hexdigest())])
a=np.array([r['nll_mean'] for r in old]);b=np.array([r['nll_mean'] for r in new]);d=b-a;rng=np.random.default_rng(20261017);boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
out=dict(time=time.time(),n=len(d),old_example_mean_nll=float(a.mean()),plain_example_mean_nll=float(b.mean()),paired_plain_minus_old_nll=float(d.mean()),question_ci95=np.quantile(boot,[.025,.975]).tolist(),nll_lower=int((d<0).sum()),nll_higher=int((d>0).sum()),nll_equal=int((d==0).sum()),old_tokens=sum(r['tokens'] for r in old),plain_tokens=sum(r['tokens'] for r in new),old_token_mean_nll=sum(r['nll_sum'] for r in old)/sum(r['tokens'] for r in old),plain_token_mean_nll=sum(r['nll_sum'] for r in new)/sum(r['tokens'] for r in new),score_sha256=sha(root/'scores.jsonl'),interpretation='Within-question cosmetic formatting intervention changes tokenization and teacher-forced NLL. Protected mathematical spans and word/identifier/number sequence unchanged by audited transformation; known process errors preserved. This is not yet a training-accuracy effect, nor a causal mediation estimate for all teacher32-vs7B differences.')
write(S/'audits/plain_teacher_nll.json',out);print(json.dumps(out,indent=2))
