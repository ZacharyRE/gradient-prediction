from common import *
from full_gradient_table import FullGradientTable
from predictor_core import raw,weights
from types import SimpleNamespace
from itertools import combinations
import torch

torch.manual_seed(983);torch.set_num_threads(2);N,D,R=6,8,3;A=torch.randn(R,D);B=torch.randn(D,R);mod=SimpleNamespace(lora_A={'default':SimpleNamespace(weight=A)},lora_B={'default':SimpleNamespace(weight=B)},scaling={'default':.7});s=.7
rows=[dict(index=i,x=torch.randn(5,D),g=torch.randn(5,D)) for i in range(N)];checks=[]
for mode in ['activation','factor']:
    table=FullGradientTable.__new__(FullGradientTable);table.rows=[None]*N;table.ages=[None]*N;table.known=0;table.mode=mode;table.cached=[dict(x=r['x']) for r in rows];table.total=torch.zeros(D,D,dtype=torch.float64) if mode=='activation' else [torch.zeros_like(A,dtype=torch.float64),torch.zeros_like(B,dtype=torch.float64)]
    table.observe(rows,mod,0);changed=dict(rows[2],g=torch.randn_like(rows[2]['g']));table.observe([changed],mod,1)
    total=table.sum_prediction(mod);individual=[table.predict([i],mod) for i in range(N)];mean_error=max(float((v-torch.stack([r[j] for r in individual]).sum(0)).abs().max()) for j,v in enumerate(total));assert mean_error<1e-4,mean_error
    truth=[torch.randn_like(A) for _ in range(N)];estimated=[]
    for ix in combinations(range(N),2):estimated.append(total[0]+(N/2)*(torch.stack([truth[i] for i in ix]).sum(0)-table.predict(list(ix),mod)[0]))
    error=float((torch.stack(estimated).mean(0)-torch.stack(truth).sum(0)).abs().max());assert error<1e-4,error;checks.append(dict(mode=mode,maintained_mean_max_error=mean_error,exact_enumeration_unbiased_error=error))
report=dict(passed=True,checks=checks,scope='Full-data estimator before clipping/Adam; activation transport and quantized flat-gradient memory, including overwrite update')
write(EXP/'results/full_gradient_table_cpu_audit.json',report);log(report)
