"""Small state-conditioned surrogate for a teacher's forced-format ABC logits.

This predicts behavior, not free-generation correctness. No gold-class identity
is part of the feature vector. The class-token position is a format position;
evaluation uses the identical fixed answer template for every example.
"""
import torch
from torch import nn
from torch.nn import functional as F

class BehaviorSurrogate(nn.Module):
    def __init__(self,dim=1536,width=256,class_tokens=None,delta_features=True):
        super().__init__();self.cfg=dict(dim=dim,width=width,class_tokens=class_tokens,delta_features=delta_features);self.dim=dim;self.delta_features=delta_features
        self.register_buffer('class_tokens',torch.tensor(class_tokens,dtype=torch.long))
        feature_dim=(6 if delta_features else 3)*dim
        self.register_buffer('feature_mean',torch.zeros(feature_dim));self.register_buffer('feature_scale',torch.ones(feature_dim))
        self.net=nn.Sequential(nn.Linear(feature_dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3))

    def features(self,d,A,B,s):
        h=d['h'].float();x=d['x'].float();n,L,dim=h.shape;pos=torch.arange(L,device=h.device)[None,:]
        first=d['active'].long().argmax(1);class_mask=(d['label_ids'][...,None]==self.class_tokens[None,None,:]).any(-1)
        assert torch.all(class_mask.sum(1)==1)
        where=class_mask.long().argmax(1);ii=torch.arange(n,device=h.device);prompt=(pos<=first[:,None])&d['valid']
        def pool(z):return [(z*prompt[...,None]).sum(1)/prompt.sum(1,keepdim=True),z[ii,first],z[ii,where]]
        deltas=[s*((z@A.T)@B.T) for z in pool(x)]
        pieces=[F.layer_norm(z-delta,(dim,)) for z,delta in zip(pool(h),deltas)]
        if self.delta_features:pieces+=deltas
        return torch.cat(pieces,1)

    def forward(self,features):
        logits=self.net((features-self.feature_mean)/self.feature_scale)
        return logits-logits.mean(-1,keepdim=True)

class AnchoredBehaviorSurrogate(BehaviorSurrogate):
    """Predict only the change from a paid initial-teacher class-logit readout."""
    requires_initial_teacher=True
    def __init__(self,dim=1536,width=128,class_tokens=None,delta_features=True,kind='anchored_behavior'):
        super().__init__(dim,width,class_tokens,True)
        self.cfg.update(kind='anchored_behavior');fd=6*dim+3
        self.feature_mean=torch.zeros(fd);self.feature_scale=torch.ones(fd)
        self.net=nn.Sequential(nn.Linear(fd,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3,bias=False))
        nn.init.zeros_(self.net[-1].weight)

    def inputs(self,current,initial,initial_logits):
        assert current.shape[-1]==initial.shape[-1]==6*self.dim
        return torch.cat([initial[...,:3*self.dim],current[...,3*self.dim:]-initial[...,3*self.dim:],initial_logits],-1)

    def forward(self,current,initial,initial_logits):
        x=self.inputs(current,initial,initial_logits);zero=self.inputs(initial,initial,initial_logits)
        correction=self.net((x-self.feature_mean)/self.feature_scale)-self.net((zero-self.feature_mean)/self.feature_scale)
        correction=correction-correction.mean(-1,keepdim=True)
        return initial_logits+correction
