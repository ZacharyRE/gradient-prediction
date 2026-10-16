"""Check causality and no label leakage in the token-loss surrogate."""
from common import *
from token_critic import TokenCritic
import torch

torch.manual_seed(19);torch.set_num_threads(2)
p=TokenCritic(16,width=16,depth=1,label_tokens=[1,2],pretrained_readout=False);p.eval()
h=torch.randn(1,6,16);d=dict(h=h,valid=torch.ones(1,6,dtype=torch.bool),pos=torch.arange(6).float()[None,:]/5,active=torch.tensor([[0,0,1,0,0,0]],dtype=torch.bool),label_ids=torch.tensor([[-100,-100,1,-100,-100,-100]]))
v,g=p.values_and_grad(d);changed=h.clone();changed[:,3:]=torch.randn_like(changed[:,3:])*100
with torch.no_grad():v2=p.loss_values(d,changed)
assert torch.allclose(v,v2,atol=1e-7,rtol=1e-7),(v,v2)
assert float(g[:,3:].abs().max())==0
direction=torch.randn_like(h);direction/=direction.norm();eps=.01
with torch.no_grad():fd=(p.loss_values(d,h+eps*direction).sum()-p.loss_values(d,h-eps*direction).sum())/(2*eps)
error=float(abs(fd-(g*direction).sum()));assert error<1e-4,error
p.train();v,g=p.values_and_grad(d,True);(v.square().mean()+g.square().mean()).backward()
assert all(w.grad is None or torch.isfinite(w.grad).all() for w in p.parameters())
report=dict(passed=True,future_hidden_states_cannot_change_earlier_prediction=True,future_input_gradient_zero=True,finite_difference_absolute_error=error,double_backprop_finite=True)
write(EXP/'results/token_critic_cpu_audit.json',report);log(report)
