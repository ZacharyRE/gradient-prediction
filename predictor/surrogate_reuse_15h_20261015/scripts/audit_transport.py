"""Independent CPU verification of exact gradient-factor transport algebra."""
from common import *
import torch,itertools
torch.manual_seed(44);dtype=torch.float64
x=torch.randn(5,7,9,dtype=dtype);g=torch.randn(5,7,8,dtype=dtype);A=torch.randn(3,9,dtype=dtype);B=torch.randn(8,3,dtype=dtype)
C=g.transpose(-1,-2)@x
direct_a=(g@B).transpose(-1,-2)@x;direct_b=g.transpose(-1,-2)@(x@A.T)
assert torch.allclose(direct_a,B.T@C,atol=1e-12,rtol=1e-12)
assert torch.allclose(direct_b,C@A.T,atol=1e-12,rtol=1e-12)
# Any fixed predictor, including transported class means, is unbiased with a
# uniformly sampled correction; verify all 2-of-5 subsets.
truth=torch.cat([direct_a.flatten(1),direct_b.flatten(1)],1);pred=torch.randn_like(truth)
est=[]
for ids in itertools.combinations(range(5),2):est.append(pred.sum(0)+2.5*(truth[list(ids)]-pred[list(ids)]).sum(0))
error=float((torch.stack(est).mean(0)-truth.sum(0)).abs().max());assert error<1e-12
report=dict(passed=True,transport_exact=True,raw_cv_expectation_max_error=error,adam_unbiasedness_not_claimed=True)
write(EXP/'results/transport_cpu_audit.json',report);log(report)
