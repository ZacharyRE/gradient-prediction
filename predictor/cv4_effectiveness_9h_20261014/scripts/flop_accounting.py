"""Cover fused PyTorch encoder inference without changing the measured execution path."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

class _GlobalTracker:
 # ModuleTracker installs autograd hooks; operator totals need no module hooks.
 parents={'Global'}
 def __enter__(self):return self
 def __exit__(self,*args):return False

class _NativeArithmeticDispatch(TorchDispatchMode):
 def __init__(self,counter):super().__init__();self.counter=counter
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {};out=func(*args,**kwargs)
  return self.counter._count_flops(getattr(func,'_overloadpacket',func),out,args,kwargs)

class NativeFlopCounterMode(FlopCounterMode):
 """Use Torch's formulas without decomposing native ops or adding gradient hooks."""
 def __enter__(self):
  self.flop_counts.clear();self.mod_tracker=_GlobalTracker();self.mod_tracker.__enter__()
  self.mode=_NativeArithmeticDispatch(self);self.mode.__enter__();return self

def fused_encoder_flops(src,embed_dim,num_heads,qkv_weight,qkv_bias,proj_weight,proj_bias,use_gelu,norm_first,eps,norm_weight_1,norm_bias_1,norm_weight_2,norm_bias_2,ffn_weight_1,ffn_bias_1,ffn_weight_2,ffn_bias_2,*args,**kwargs):
 b,l,d=src;h=ffn_weight_1[0]
 assert d==embed_dim and tuple(qkv_weight)==(3*d,d) and tuple(proj_weight)==(d,d) and tuple(ffn_weight_2)==(d,h)
 # Four attention projections, QK^T and AV, and the two feed-forward projections.
 # Dense padded execution; multiply-add=2. Excludes normalization/softmax/nonlinearity.
 return 8*b*l*d*d+4*b*l*l*d+4*b*l*d*h

def fused_mha_flops(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,*args,**kwargs):
 b,l,d=query
 assert tuple(query)==tuple(key)==tuple(value) and d==embed_dim
 assert tuple(qkv_weight)==(3*d,d) and tuple(proj_weight)==(d,d)
 return 8*b*l*d*d+4*b*l*l*d

FUSED_MAPPING={torch.ops.aten._transformer_encoder_layer_fwd:fused_encoder_flops,torch.ops.aten._native_multi_head_attention:fused_mha_flops}

class OperatorCoverage(TorchDispatchMode):
 def __init__(self):super().__init__();self.seen=set()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.seen.add(getattr(func,'_overloadpacket',func))
  return func(*args,**(kwargs or {}))

def coverage_record(coverage,counter):
 unregistered=coverage.seen-set(counter.flop_registry)
 opaque_attention=[str(op) for op in unregistered if any(k in str(op) for k in ['_transformer_encoder','_native_multi_head','_scaled_dot_product','_flash_attention','_efficient_attention'])]
 assert not opaque_attention,('Uncounted opaque attention operations',opaque_attention)
 return dict(seen_operations=sorted(str(op) for op in coverage.seen),operations_without_flop_formula=sorted(str(op) for op in unregistered),custom_formula_operations=sorted(str(op) for op in coverage.seen if op in FUSED_MAPPING),scope='Fused dense encoder/MHA matrix arithmetic counted by explicit shape formulas; scalar, normalization, activation, softmax, copies, and optimizer operations remain excluded. Unsupported opaque attention operations cause failure.')
