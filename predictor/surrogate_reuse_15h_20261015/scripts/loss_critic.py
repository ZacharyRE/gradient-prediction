"""A scalar suffix-loss surrogate; its input derivative supplies synthetic gradients."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

class LossCritic(nn.Module):
    def __init__(self,dim,width=128,depth=2,label_tokens=None,kind='loss_critic'):
        super().__init__();self.cfg=dict(kind=kind,dim=dim,width=width,depth=depth,label_tokens=label_tokens)
        self.input_field='h';self.register_buffer('target_scale',torch.tensor(1.));self.register_buffer('loss_scale',torch.tensor(1.))
        self.norm=nn.LayerNorm(dim);self.proj=nn.Linear(dim,width);self.extra=nn.Linear(5,width,bias=False)
        tokens=label_tokens or [];self.register_buffer('label_tokens',torch.tensor(tokens,dtype=torch.long));self.label_emb=nn.Embedding(len(tokens)+1,width)
        self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,0.,'gelu',batch_first=True,norm_first=True),depth,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        # Higher derivatives require the mathematical attention implementation.
        torch.backends.mha.set_fastpath_enabled(False)

    def loss_values(self,d,h):
        pos=d['pos'];rms=h.square().mean(-1).clamp_min(1e-12).sqrt().log()
        label_index=torch.zeros_like(d['label_ids'])
        for i,token in enumerate(self.label_tokens):label_index=torch.where(d['label_ids']==token,i+1,label_index)
        z=self.proj(self.norm(h))+self.extra(torch.stack([rms,d['active'].float(),pos,pos.square(),torch.sin(torch.pi*pos)],-1))+self.label_emb(label_index)
        with sdpa_kernel(SDPBackend.MATH):z=self.body(z,src_key_padding_mask=~d['valid'])
        per_token=F.softplus(self.head(z).squeeze(-1))*self.loss_scale
        return (per_token*d['active']).sum(1)

    def values_and_grad(self,d,create_graph=False):
        with torch.enable_grad(),torch.autocast('cuda',enabled=False):
            h=d['h'].detach().float().requires_grad_(True)
            values=self.loss_values(d,h)
            grad=torch.autograd.grad(values.sum(),h,create_graph=create_graph,retain_graph=create_graph)[0]
        return values,grad

    def forward(self,d):
        _,g=self.values_and_grad(d,create_graph=self.training)
        return g if self.training else g.detach()
