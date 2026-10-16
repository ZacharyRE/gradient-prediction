"""Check table transport, stale/fresh separation, and finite-subset unbiasedness."""
from common import *
from gradient_table import GradientTable,raw
from correction4 import corrected_pair
from types import SimpleNamespace
from itertools import combinations
import torch

torch.manual_seed(81);torch.set_num_threads(2)
A=torch.randn(3,7);B=torch.randn(7,3);mod=SimpleNamespace(lora_A={'default':SimpleNamespace(weight=A)},lora_B={'default':SimpleNamespace(weight=B)},scaling={'default':.7})
rows=[dict(index=i,x=torch.randn(5+i,7).bfloat16(),g=torch.randn(5+i,7)) for i in range(5)]
table=GradientTable(None,None,mod,None,list(range(5)),rows,{})
table.observe(rows[:3],mod,2)
A.add_(.2);B.mul_(1.2)
pred=table.predict([0,1,2,3,4],mod)
for i in range(3):
 expected=raw(rows[i]['x'][None],rows[i]['g'].bfloat16()[None],A,B,.7)
 assert all(torch.allclose(p[i],e[0],atol=1e-5,rtol=1e-5) for p,e in zip(pred,expected))
assert not pred[0][3:].any() and not pred[1][3:].any()
truth=[torch.randn_like(p) for p in pred];averages=[torch.zeros_like(p[0]) for p in pred];samples=list(combinations(range(5),2))
for subset in samples:
 estimate=corrected_pair(pred,[t[list(subset)].sum(0) for t in truth],list(subset),10)
 for a,e in zip(averages,estimate):a.add_(e/len(samples))
error=max(float((a-t.sum(0)/10).abs().max()) for a,t in zip(averages,truth));assert error<2e-6,error
report=dict(passed=True,activation_memory_uses_current_factors=True,unseen_examples_predict_zero=True,before_clipping_adam_unbiased_max_absolute_error=error,table_update_affects_future_estimators_only='Enforced by step construction; current gradients materialized before observe')
write(EXP/'results/gradient_table_cpu_audit.json',report);log(report)
