"""Descriptive length/truncation audits; post-treatment strata are not causal estimates."""
from common import *
import numpy as np,csv
root=S/'results/evaluation/dev';base=read(root/'base/math_predictions.jsonl');out=[]
for p in sorted(root.glob('*/math_predictions.jsonl')):
 if not (p.parent/'math_summary.json').exists():continue
 rr=read(p);assert len(rr)==len(base);assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
 lengths=np.array([r['generated_tokens'] for r in rr]);bt=np.array([r['finish_reason']=='length' for r in base]);mt=np.array([r['finish_reason']=='length' for r in rr]);bc=np.array([r['correct'] for r in base]);mc=np.array([r['correct'] for r in rr]);both=~bt&~mt
 out.append(dict(name=p.parent.name,correct=int(mc.sum()),mean_tokens=float(lengths.mean()),median_tokens=float(np.median(lengths)),p95_tokens=float(np.quantile(lengths,.95)),truncated=int(mt.sum()),baseline_truncated=int(bt.sum()),base_truncated_model_not=int((bt&~mt).sum()),model_truncated_base_not=int((mt&~bt).sum()),wins=int((mc&~bc).sum()),losses=int((bc&~mc).sum()),both_untruncated_n=int(both.sum()),both_untruncated_wins=int((both&mc&~bc).sum()),both_untruncated_losses=int((both&bc&~mc).sum())))
write(S/'audits/decoding_length_audit.json',dict(time=time.time(),rows=out,limitation='Length/termination strata are observed after treatment and not randomized or causal controls. A truncation upper bound is not an achievable improvement.'))
with (S/'audits/decoding_length_audit.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
sample=S/'results/sampling/student_epoch2_pilot';bb=read(sample/'base/predictions.jsonl');rr=read(sample/'sample_all_lr1e5_s43_epoch2/predictions.jsonl');stats={}
for name,data in [('base',bb),('sample43ep2',rr)]:
 flat=[x for r in data for x in r['samples']];stats[name]=dict(n=len(flat),correct=sum(r['correct'] for r in flat),truncated=sum(r['finish_reason']=='length' for r in flat),mean_tokens=float(np.mean([r['generated_tokens'] for r in flat])))
write(S/'audits/sampling_length_audit.json',dict(time=time.time(),stats=stats,limitation='Common decoding seeds do not imply sample-index-matched reasoning trajectories. Main uncertainty clusters by question, not by treating2048 answers as independent questions.'))
for r in out:
 if r['name'] in ['base','sample_all_lr1e5_s43_epoch2','teacher_all_lr5e5_s43_epoch4']:print(json.dumps(r))
print(json.dumps(stats))
