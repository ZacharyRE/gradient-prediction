"""Verify compact fitting objective equals actual LoRA-factor squared error."""
from common import *
import torch
torch.manual_seed(71);torch.set_num_threads(2);dtype=torch.float64
U=torch.linalg.qr(torch.randn(8,3,dtype=dtype)).Q;V=torch.linalg.qr(torch.randn(9,4,dtype=dtype)).Q
A=torch.randn(2,9,dtype=dtype);B=torch.randn(8,2,dtype=dtype);C=torch.randn(5,3,4,dtype=dtype,requires_grad=True);ga=torch.randn(5,2,9,dtype=dtype);gb=torch.randn(5,8,2,dtype=dtype);s=.7
M=B.T@U;N=V.T@A.T;na=ga.square().sum((1,2)).mean();nb=gb.square().sum((1,2)).mean()
full=.5*(((s*(M@C@V.T)-ga).square().sum((1,2))/na)+((s*(U@C@N)-gb).square().sum((1,2))/nb)).mean()
cross=(M.T@(ga@V))/na+((U.T@gb)@N.T)/nb
compact=.5*(s*s*((C*((M.T@M/na)@C)).sum((1,2))+(C*(C@(N@N.T/nb))).sum((1,2)))-2*s*(C*cross).sum((1,2))+ga.square().sum((1,2))/na+gb.square().sum((1,2))/nb).mean()
error=float(abs(full-compact).detach());grad_error=float((torch.autograd.grad(full,C,retain_graph=True)[0]-torch.autograd.grad(compact,C)[0]).abs().max());assert error<1e-12 and grad_error<1e-12
report=dict(passed=True,objective_absolute_error=error,gradient_max_absolute_error=grad_error);write(EXP/'results/factor_loss_cpu_audit.json',report);log(report)
