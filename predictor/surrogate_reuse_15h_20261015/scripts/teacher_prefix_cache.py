"""Cache the invariant prefix for single-layer LoRA, for teacher and CV alike.

Store the exact residual input to layer8 (original dtype) and attention output
projection input. Recompute the trainable projection and entire suffix. Full
input masks/positions are still constructed by the native Qwen model. Cache
construction batch shapes can change BF16 rounding; audit before comparison.
"""
import types,time
import predictor_core as pc
from predictor_core import torch

class TeacherPrefixCache:
    def __init__(self,m,tok,mod,cap,ds,ids=None,micro=32):
        self.m=m;self.ds=ds;self.mod=mod;self.enabled=False;self.active=None
        self.layers=m.get_base_model().model.layers;self.saved=[];self.rows={}
        assert all(('layers.8.self_attn.o_proj.lora_' in name) for name,w in m.named_parameters() if w.requires_grad)
        captured={}
        def residual_hook(module,args):captured['residual']=args[0].detach()
        class PrefixDone(Exception):pass
        def stop(*args):raise PrefixDone()
        h1=self.layers[8].register_forward_pre_hook(residual_hook)
        h2=mod.register_forward_hook(stop)
        torch.cuda.synchronize();start=time.perf_counter()
        try:
            with torch.no_grad():
                for ix in pc.chunks(ds,list(range(len(ds))) if ids is None else ids,max_n=micro):
                    bb=pc.batch(ds,ix,tok)
                    try:pc.forward(m,bb)
                    except PrefixDone:pass
                    for j,i in enumerate(ix):
                        length=len(ds[i]['input_ids'])
                        self.rows[i]=dict(x=cap.x[j,:length].detach().clone(),residual=captured['residual'][j,:length].clone())
        finally:h1.remove();h2.remove()
        torch.cuda.synchronize();self.seconds=time.perf_counter()-start
        self.bytes=sum(v.numel()*v.element_size() for row in self.rows.values() for v in row.values())
        self.original_batch=pc.batch
        original_batch=self.original_batch
        def cached_batch(dataset,indices,tokenizer,*args,**kwargs):
            b=original_batch(dataset,indices,tokenizer,*args,**kwargs)
            self.active=None
            if self.enabled and dataset is self.ds:
                length=b['input_ids'].shape[1];rows=[self.rows[i] for i in indices]
                self.active={}
                for key in ['x','residual']:
                    v=rows[0][key];packed=torch.zeros(len(rows),length,v.shape[-1],device=v.device,dtype=v.dtype)
                    for j,row in enumerate(rows):packed[j,:len(row[key])]=row[key]
                    self.active[key]=packed
            return b
        self.batch_wrapper=cached_batch
        for module in [pc]:
            self.saved.append((module,'batch',module.batch));module.batch=cached_batch
        for index in range(9):
            layer=self.layers[index];original=layer.forward
            if index<8:
                def forward(layer,hidden_states,*args,_original=original,**kwargs):
                    return hidden_states if self.enabled and self.active is not None else _original(hidden_states,*args,**kwargs)
            else:
                def forward(layer,hidden_states,*args,_original=original,**kwargs):
                    if not self.enabled or self.active is None:return _original(hidden_states,*args,**kwargs)
                    assert not kwargs.get('use_cache',False),'Training cache does not support autoregressive decoding'
                    h=self.active['residual']+self.mod(self.active['x'])
                    return h+layer.mlp(layer.post_attention_layernorm(h))
            self.saved.append((layer,'forward',original));layer.forward=types.MethodType(forward,layer)
        self.enabled=True

    def close(self):
        self.enabled=False;self.active=None
        for obj,key,original in reversed(self.saved):setattr(obj,key,original)
        self.saved=[]
