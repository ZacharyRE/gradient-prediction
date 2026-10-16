from common import *
from class_covariance import FrozenClassCovariance,StratifiedGradients
from factor_surrogate import FactorSurrogate
from types import SimpleNamespace
from itertools import combinations,product
import torch

torch.manual_seed(61);torch.set_num_threads(2);A=torch.randn(3,8);B=torch.randn(8,3);mod=SimpleNamespace(lora_A={'default':SimpleNamespace(weight=A)},lora_B={'default':SimpleNamespace(weight=B)},scaling={'default':.7})
f=FrozenClassCovariance.__new__(FrozenClassCovariance);f.U=torch.randn(8,4);f.V=torch.randn(8,5);f.means=torch.randn(3,4,5);f.classes=np.array([0,0,0,1,1,2]);ids=list(range(6));positions=[0,3,5]
a,b=f.correction(mod,ids,positions);C=f.means[torch.tensor(f.classes)];pa=.7*((B.T@f.U)[None]@C@f.V.T);pb=.7*(f.U[None]@C@(f.V.T@A.T));expected=[p.sum(0)-2*p[positions].sum(0) for p in [pa,pb]];error=max(float((x-y).abs().max()) for x,y in zip([a,b],expected));assert error<1e-4,error
# Exact enumeration for a fixed proportional allocation, not a Monte Carlo test.
s=StratifiedGradients(SimpleNamespace(rows=[{'gold':'ABC'[c]} for c in f.classes]),123);pos,w=s.sample(ids,4);allocation=np.bincount(f.classes[pos],minlength=3);g=np.random.default_rng(43).normal(size=(6,9));est=[]
for parts in product(*[list(combinations(np.flatnonzero(f.classes==c),int(allocation[c]))) for c in range(3)]):est.append(sum(len(np.flatnonzero(f.classes==c))/allocation[c]*g[list(ix)].sum(0) for c,ix in enumerate(parts)))
unbiased_error=float(np.abs(np.mean(est,0)-g.sum(0)).max());assert unbiased_error<1e-12,unbiased_error
report=dict(passed=True,class_count_correction_matches_per_example_matrix_prediction_max_error=error,stratified_weighted_gradient_exact_expectation_error=unbiased_error,scope='Mathematical estimator before BF16 rounding, clipping and Adam')
write(EXP/'results/class_controls_cpu_audit.json',report);log(report)
