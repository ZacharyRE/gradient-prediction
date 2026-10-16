"""Causal token-probability surrogate with a cross-entropy-derived gradient.

Unlike a free scalar loss critic, the network does not receive the target label
as an input feature. Labels only enter cross entropy. The vocabulary is the
union of supervised answer tokens; omitted-vocabulary mass is an approximation.
True-gradient CV corrections still use the original full-vocabulary objective.
"""
from loss_critic import *
from pathlib import Path

class TokenCritic(LossCritic):
    def __init__(self,dim,width=128,depth=2,label_tokens=None,kind='token_critic',pretrained_readout=True):
        super().__init__(dim,width,depth,label_tokens)
        self.cfg=dict(kind=kind,dim=dim,width=width,depth=depth,label_tokens=label_tokens,pretrained_readout=pretrained_readout)
        self.pretrained_readout=pretrained_readout
        del self.label_emb
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,len(label_tokens)))
        self.register_buffer('readout',torch.zeros(len(label_tokens),dim));self.register_buffer('readout_norm',torch.ones(dim))
        self.readout_logscale=nn.Parameter(torch.zeros(()))
        if pretrained_readout:nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    @torch.no_grad()
    def initialize_readout(self,model_path):
        if not self.pretrained_readout:return
        from safetensors import safe_open
        tensors={}
        for path in Path(model_path).glob('*.safetensors'):
            with safe_open(str(path),framework='pt',device='cpu') as f:
                keys=set(f.keys())
                for key in ['lm_head.weight','model.embed_tokens.weight','model.norm.weight']:
                    if key in keys:
                        t=f.get_tensor(key)
                        tensors[key]=t[self.label_tokens.cpu()] if key.endswith('tokens.weight') or key=='lm_head.weight' else t
        self.readout.copy_(tensors.get('lm_head.weight',tensors['model.embed_tokens.weight']));self.readout_norm.copy_(tensors['model.norm.weight'])

    def token_logits(self,d,h):
        pos=d['pos'];rms=h.square().mean(-1).clamp_min(1e-12).sqrt().log()
        z=self.proj(self.norm(h))+self.extra(torch.stack([rms,d['active'].float(),pos,pos.square(),torch.sin(torch.pi*pos)],-1))
        L=h.shape[1];causal=torch.ones(L,L,device=h.device,dtype=torch.bool).triu(1)
        with sdpa_kernel(SDPBackend.MATH):z=self.body(z,mask=causal,src_key_padding_mask=~d['valid'],is_causal=True)
        logits=self.head(z)
        if self.pretrained_readout:
            normalized=h*torch.rsqrt(h.square().mean(-1,keepdim=True)+1e-6)*self.readout_norm
            logits=logits+self.readout_logscale.exp()*(normalized@self.readout.T)
        return logits

    def loss_values(self,d,h):
        logits=self.token_logits(d,h)
        targets=torch.full_like(d['label_ids'],-100)
        for i,token in enumerate(self.label_tokens):targets=torch.where(d['label_ids']==token,i,targets)
        assert torch.equal(targets>=0,d['active'])
        losses=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none',ignore_index=-100).reshape_as(targets)
        return losses.sum(1)
