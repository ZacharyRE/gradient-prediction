"""Same assistant-token CE objective, computing vocabulary logits only where supervised.

This backend is applied to both oracle and predictor correction labels. Full input
sequences and the complete frozen transformer are still evaluated.
"""
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import predictor_core as pc

ORIGINAL_FORWARD=pc.forward
ORIGINAL_CHUNKS=pc.chunks
TOKEN_BUDGET=2048

def supervised_forward(m,b):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
        base=m.get_base_model()
        out=base.model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False,return_dict=True)
        shifted=b['labels'][:,1:]
        valid=shifted!=-100
        selected=out.last_hidden_state[:,:-1][valid]
        logits=base.lm_head(selected)
        loss=F.cross_entropy(logits.float(),shifted[valid],reduction='mean')
        return SimpleNamespace(loss=loss)

def chunks(ds,ids,max_n=4,token_budget=2048):
    return ORIGINAL_CHUNKS(ds,ids,max_n=max_n,token_budget=TOKEN_BUDGET)

def configure(enabled=False,token_budget=2048):
    global TOKEN_BUDGET
    TOKEN_BUDGET=token_budget
    pc.forward=supervised_forward if enabled else ORIGINAL_FORWARD
    pc.chunks=chunks
