"""CPU audit of gradients, higher derivatives and inference under no_grad."""
from common import *
import torch
from loss_critic import LossCritic

torch.manual_seed(7);torch.set_num_threads(2)
p=LossCritic(16,width=16,depth=1,label_tokens=[1,2]);p.eval()
h=torch.randn(2,6,16);valid=torch.ones(2,6,dtype=torch.bool);valid[1,4:]=False
d=dict(h=h,valid=valid,pos=torch.arange(6).float()[None,:].expand(2,-1)/5,active=torch.tensor([[0,0,0,1,1,1],[0,0,1,1,0,0]],dtype=torch.bool),label_ids=torch.tensor([[-100,-100,-100,1,2,1],[-100,-100,1,2,-100,-100]]))
v,g=p.values_and_grad(d);direction=torch.randn_like(h);direction=direction/direction.norm();epsilon=.01
with torch.no_grad():fd=(p.loss_values(d,h+epsilon*direction).sum()-p.loss_values(d,h-epsilon*direction).sum())/(2*epsilon)
ad=(g*direction).sum();error=float(abs(fd-ad));assert error<1e-4,(fd,ad,error)
assert float(g[1,4:].abs().max())==0
with torch.no_grad():gg=p(d)
assert torch.equal(g,gg)
p.train();v,g=p.values_and_grad(d,True);loss=v.square().mean()+g.square().mean();loss.backward()
assert all(w.grad is None or torch.isfinite(w.grad).all() for w in p.parameters())
report=dict(passed=True,finite_difference_absolute_error=error,padding_gradient_zero=True,no_grad_inference_consistent=True,double_backprop_finite=True)
write(EXP/'results/loss_critic_cpu_audit.json',report);log(report)
