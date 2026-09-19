"""Single-module Countdown synthetic gradients. No base-model serialization."""
import sys, json, random, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
OLD = ROOT / 'research/0.5B_countdown_lora'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD / 'scripts'))
from common import MODEL as SHARED_MODEL, prompt, read, write, jsonl
from sft_helpers import CompletionDataset, Collator
from countdown import check, number_key
MODEL=json.loads((EXP/'configs/protocol.json').read_text()).get('base_model_reference',SHARED_MODEL)

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True

def setup_model(adapter=None, layer=8, rank=64, rng=101):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, PeftModel
    seed(rng)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True).cuda()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank,
            lora_dropout=0, target_modules=[f'model.layers.{layer}.self_attn.o_proj'],
            task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache=False
    mods = [(n,m) for n,m in model.named_modules() if hasattr(m,'lora_A')]
    assert len(mods)==1 and mods[0][0].endswith(f'layers.{layer}.self_attn.o_proj')
    assert all(('lora_' in n)==p.requires_grad for n,p in model.named_parameters())
    return model, tok, mods[0][1]

def batch(ds, ix, tok):
    return {k:v.cuda() for k,v in Collator(tok.pad_token_id)([ds[i] for i in ix]).items()}

def forward(model,b):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return model(**b, use_cache=False)

def weights(mod):
    return mod.lora_A['default'].weight, mod.lora_B['default'].weight, mod.scaling['default']

def raw_grads(x,g,A,B,scale=1):
    """x [T,in], g [T,out]; returns full raw A/B gradient, no sketch."""
    x=x.float();g=g.float()
    return scale*((g@B.float()).T@x), scale*(g.T@(x@A.float().T))

def flat(pair): return torch.cat([v.flatten() for v in pair])

def measures(pred,true):
    p=pred.flatten().float();t=true.flatten().float()
    return dict(cosine=float(F.cosine_similarity(p,t,dim=0)),
        relative_l2=float((p-t).norm()/t.norm().clamp_min(1e-12)),
        norm_ratio=float(p.norm()/t.norm().clamp_min(1e-12)))

class Capture:
    def __init__(self, model, mod, layer):
        self.x=self.y=self.z=self.residual=None
        def hook(m,args,out):
            self.x=args[0].detach();self.y=out
            if out.requires_grad:out.retain_grad()
        def block(m,args,out):
            self.z_graph=out[0] if isinstance(out,tuple) else out
            self.z=self.z_graph.detach()
        def pre(m,args):self.residual=args[0].detach()
        self.handles=[model.base_model.model.model.layers[layer].register_forward_pre_hook(pre),mod.register_forward_hook(hook),
            model.base_model.model.model.layers[layer].register_forward_hook(block)]
    def close(self):
        for h in self.handles:h.remove()

def data_path(name): return EXP/'data'/f'{name}.jsonl'

