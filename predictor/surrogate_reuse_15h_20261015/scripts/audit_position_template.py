from common import *
from position_template import PositionTemplate
from predictor_core import raw
import torch

torch.manual_seed(977);torch.set_num_threads(2);n,L,D=3,11,8;A=torch.randn(3,D);B=torch.randn(D,3);labels=torch.full((n,L),-100);labels[:,4:10]=torch.tensor([5,6,7,2,8,9]);labels[1,7]=3;labels[2,7]=4
d=dict(x=torch.randn(n,L,D),active=labels>=0,label_ids=labels,valid=torch.ones(n,L,dtype=torch.bool));checks=[]
for bins in [0,1,4,8]:
    p=PositionTemplate(D,[2,3,4],bins=bins);p.templates.normal_();dense=torch.zeros(n,L,D)
    for i in range(n):
        if bins:
            for b in range(bins):
                ids=[j for j in range(4) if j*bins//4==b]
                if ids:dense[i,ids]=p.templates[i,b]/len(ids)
        dense[i,4:10]=p.templates[i,bins:]
    pp=p.predict_factors(d,A,B,.7);tt=raw(d['x'],dense,A,B,.7);error=max(float((a-b).abs().max()) for a,b in zip(pp,tt));assert error<1e-4,error;checks.append(dict(bins=bins,maximum_error=error))
report=dict(passed=True,checks=checks,scope='Compact X averages and template gradient masses match dense activation-gradient reconstruction, including empty bins')
write(EXP/'results/position_template_cpu_audit.json',report);log(report)
