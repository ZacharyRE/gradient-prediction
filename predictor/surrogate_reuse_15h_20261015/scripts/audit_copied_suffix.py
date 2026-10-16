from common import *
from copied_suffix import CopiedSuffix
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
import torch

torch.manual_seed(991);torch.set_num_threads(2)
config=Qwen2Config(hidden_size=16,intermediate_size=32,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8).to_dict();p=CopiedSuffix(16,[0,1,2],config,source_layers=[1,3],prefix_layer=0)
with torch.no_grad():p.readout.normal_()
h=torch.randn(2,5,16);labels=torch.tensor([[-100,-100,0,1,2],[-100,-100,0,1,2]]);d=dict(h=h,valid=torch.ones(2,5,dtype=torch.bool),active=labels>=0,label_ids=labels)
p.eval();z=p.token_logits(d,h);changed=h.clone();changed[:,4]+=torch.randn(2,16)*10;z2=p.token_logits(d,changed);causal=float((z[:,:4]-z2[:,:4]).abs().max());assert causal==0
values,g=p.values_and_grad(d,True);loss=g.square().mean();loss.backward();assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
direction=torch.randn_like(h);direction/=direction.norm();eps=.002
with torch.no_grad():fd=float((p.loss_values(d,h+eps*direction).sum()-p.loss_values(d,h-eps*direction).sum())/(2*eps));analytic=float((g.detach()*direction).sum())
relative=abs(fd-analytic)/max(abs(analytic),1e-4);assert relative<.02,(fd,analytic,relative)
report=dict(passed=True,causality_max_error=causal,directional_derivative_relative_error=relative,second_derivative_parameter_gradients_finite=True,scope='Small CPU Qwen configuration; actual pretrained initialization and GPU fitting separately checked')
write(EXP/'results/copied_suffix_cpu_audit.json',report);log(report)
