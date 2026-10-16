from common import *
from tangent_suffix import TangentSuffix
from types import SimpleNamespace
import torch
from torch.nn import functional as F

torch.manual_seed(93);torch.set_num_threads(2);dim=7;rank=3;s=.7
A=torch.randn(rank,dim,requires_grad=True);B=torch.randn(dim,rank,requires_grad=True);mod=SimpleNamespace(lora_A={'default':SimpleNamespace(weight=A)},lora_B={'default':SimpleNamespace(weight=B)},scaling={'default':s})
t=TangentSuffix.__new__(TangentSuffix);t.A0=A.detach().clone()*.8;t.B0=B.detach().clone()*.9;t.rows={};t.cache=SimpleNamespace(rows={})
for i in range(4):
 L=5+i;t.cache.rows[i]=dict(x=torch.randn(L,dim));t.rows[i]=dict(length=L,jac=torch.randn(2,L,dim).bfloat16(),logits=torch.randn(2),cls=i%3)
pa,_=t.predict(list(range(4)),mod);loss=0.
for i in range(4):
 x=t.cache.rows[i]['x'];r=t.rows[i];delta=s*((x@A.T)@B.T-(x@t.A0.T)@t.B0.T);diff=r['logits']+(r['jac'].float()*delta[None]).sum((1,2));logits=torch.cat([diff,torch.zeros(1)]);loss+=F.cross_entropy(logits[None],torch.tensor([r['cls']]))
truth=torch.autograd.grad(loss,(A,B));errors=[float((p.sum(0)-q).abs().max()) for p,q in zip(pa,truth)];assert max(errors)<2e-4,errors
report=dict(passed=True,analytic_lora_gradients_match_autodiff_of_local_logit_surrogate=True,max_absolute_errors=errors,two_logit_differences_represent_three_class_softmax=True)
write(EXP/'results/tangent_algebra_cpu_audit.json',report);log(report)
