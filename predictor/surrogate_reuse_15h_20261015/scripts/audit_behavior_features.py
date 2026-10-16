from common import *
from behavior_surrogate import BehaviorSurrogate
import torch

torch.manual_seed(101);torch.set_num_threads(2)
p=BehaviorSurrogate(dim=8,width=16,class_tokens=[1,2,3]);A=torch.randn(2,8);B=torch.randn(8,2)
d=dict(h=torch.randn(2,9,8),x=torch.randn(2,9,8),active=torch.tensor([[0,0,0,1,1,1,1,1,0]]*2,dtype=torch.bool),valid=torch.ones(2,9,dtype=torch.bool),label_ids=torch.tensor([[-100,-100,-100,8,9,1,10,11,-100],[-100,-100,-100,8,9,2,10,11,-100]]))
a=p.features(d,A,B,1)
changed={k:v.clone() for k,v in d.items()};changed['label_ids'][:,5]=3
assert torch.equal(a,p.features(changed,A,B,1))
changed['h'][:,6:]=100*torch.randn_like(changed['h'][:,6:]);changed['x'][:,6:]=100*torch.randn_like(changed['x'][:,6:])
assert torch.equal(a,p.features(changed,A,B,1))
report=dict(passed=True,gold_class_identity_cannot_change_features=True,hidden_states_after_class_prediction_cannot_change_features=True,scope='Feature-level audit; teacher attention remains causal and evaluation uses one fixed answer template')
write(EXP/'results/behavior_features_cpu_audit.json',report);log(report)
