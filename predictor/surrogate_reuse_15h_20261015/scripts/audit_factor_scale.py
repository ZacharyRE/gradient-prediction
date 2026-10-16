from common import *
from factor_scale import FactorScale
from factor_surrogate import FactorSurrogate
import torch

torch.manual_seed(991);torch.set_num_threads(2);n,L,D=4,7,8;A=torch.randn(3,D);B=torch.randn(D,3);s=.7;tokens=[2,3,4]
p=FactorScale(D,4,5,3,tokens);q=FactorSurrogate(D,4,5,16,tokens)
with torch.no_grad():
    for name in ['U','V','class_mean']:
        getattr(p,name).normal_();getattr(q,name).copy_(getattr(p,name))
    p.feature_basis.normal_()
labels=torch.full((n,L),-100);labels[:,4]=1;labels[:,5]=torch.tensor([2,3,4,2]);labels[:,6]=5;d=dict(h=torch.randn(n,L,D),x=torch.randn(n,L,D),active=labels>=0,label_ids=labels,valid=torch.ones(n,L,dtype=torch.bool))
with torch.no_grad():actual=p.predict_factors(d,A,B,s);reference=q.predict_factors(d,A,B,s)
assert all(torch.equal(a,b) for a,b in zip(actual,reference))
x=torch.randn(50,6,dtype=torch.float64);weights=torch.rand(50,dtype=torch.float64);target=torch.randn(50,dtype=torch.float64);alpha=.1
w=torch.linalg.solve(x.T@(weights[:,None]*x)/len(x)+alpha*torch.eye(6,dtype=x.dtype),x.T@(weights*target)/len(x)).requires_grad_();loss=(weights*(x@w-target).square()).mean()+alpha*w.square().sum();grad=torch.autograd.grad(loss,w)[0];error=float(grad.abs().max());assert error<1e-12
report=dict(passed=True,unit_scale_matches_zero_head_factor_predictor_bitwise=True,ridge_stationarity_max_error=error,scope='CPU algebra; downstream and GPU timing are independent experiments')
write(EXP/'results/factor_scale_cpu_audit.json',report);log(report)
