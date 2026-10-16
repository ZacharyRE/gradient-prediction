"""Predict a compact matrix C, then analytically reconstruct current LoRA gradients."""
import torch
from torch import nn
from torch.nn import functional as F

class FactorSurrogate(nn.Module):
    def __init__(self,dim,rank_u=32,rank_v=64,width=256,class_tokens=None,kind='factor_surrogate',feature_mode='hx'):
        super().__init__();self.cfg=dict(kind=kind,dim=dim,rank_u=rank_u,rank_v=rank_v,width=width,class_tokens=class_tokens,feature_mode=feature_mode);self.feature_mode=feature_mode
        self.input_field='h';self.register_buffer('target_scale',torch.tensor(1.));self.register_buffer('U',torch.zeros(dim,rank_u));self.register_buffer('V',torch.zeros(dim,rank_v))
        self.register_buffer('class_tokens',torch.tensor(class_tokens));self.register_buffer('class_mean',torch.zeros(len(class_tokens),rank_u,rank_v));self.register_buffer('coefficient_scale',torch.ones(rank_u,rank_v))
        self.body=nn.Sequential(nn.Linear((6 if feature_mode=='hx' else 3)*dim+len(class_tokens)+1,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,rank_u*rank_v))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)

    def features(self,d):
        h=d['h'].float();batch,length,dim=h.shape;positions=torch.arange(length,device=h.device)[None,:]
        first=d['active'].long().argmax(1);class_mask=(d['label_ids'][...,None]==self.class_tokens[None,None,:]);where=class_mask.any(-1).long().argmax(1)
        assert torch.all(class_mask.any(-1).sum(1)==1),'Exactly one supervised class token expected'
        cls=class_mask.long().sum(1).argmax(1);prompt=(positions<=first[:,None])&d['valid'];mean=(h*prompt[...,None]).sum(1)/prompt.sum(1,keepdim=True)
        index=torch.arange(batch,device=h.device);pieces=[F.layer_norm(z,(dim,)) for z in [mean,h[index,first],h[index,where]]]
        if self.feature_mode=='hx':
            x=d['x'].float();xmean=(x*prompt[...,None]).sum(1)/prompt.sum(1,keepdim=True)
            pieces.extend(F.layer_norm(z,(dim,)) for z in [xmean,x[index,first],x[index,where]])
        onehot=F.one_hot(cls,len(self.class_tokens)).float();loglength=d['valid'].sum(1).float().log()[:,None]/8
        return torch.cat([*pieces,onehot,loglength],1),cls

    def coefficients(self,features,classes):
        return self.class_mean[classes]+self.coefficient_scale*self.body(features).reshape(-1,*self.coefficient_scale.shape)

    def predict_factors(self,d,A,B,s):
        features,cls=self.features(d);C=self.coefficients(features,cls)
        return s*((B.T@self.U)[None]@C@self.V.T),s*(self.U[None]@C@(self.V.T@A.T))
