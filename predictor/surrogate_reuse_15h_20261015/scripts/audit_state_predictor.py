from common import *
from predictor_core import condition_lora_state
from state_predictor import StatePredictor
import torch

torch.manual_seed(42);torch.set_num_threads(2)
p=StatePredictor(16,width=16,depth=1,label_dim=2,label_vocab=4);p.eval();p.label_table[1,0]=1;p.label_table[2,1]=1
A=torch.randn(4,16);B=torch.randn(16,4)
d=dict(x=torch.randn(2,7,16),h=torch.randn(2,7,16),valid=torch.tensor([[1,1,1,1,1,1,1],[1,1,1,1,0,0,0]],dtype=torch.bool),active=torch.tensor([[0,0,0,0,1,1,0],[0,0,1,1,0,0,0]],dtype=torch.bool),pos=torch.arange(7).float()[None,:].expand(2,-1)/6,label_ids=torch.tensor([[-100,-100,-100,-100,1,2,-100],[-100,-100,1,2,-100,-100,-100]]))
condition_lora_state(p,d,A,B,.7)
with torch.no_grad():first=p(d)
condition_lora_state(p,d,A*2,B/2,.7)
with torch.no_grad():same=p(d)
assert torch.equal(first,same)
condition_lora_state(p,d,A,B*.8,.7)
with torch.no_grad():changed=p(d)
assert not torch.allclose(first,changed)
assert not changed[~d['valid']].any()
p.train();p(d).square().mean().backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in p.parameters())
report=dict(passed=True,activation_prediction_invariant_to_equivalent_lora_rescaling=True,current_lora_contribution_changes_prediction=True,padded_predictions_zero=True,training_gradients_finite=True)
write(EXP/'results/state_predictor_cpu_audit.json',report);log(report)
