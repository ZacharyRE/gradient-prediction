"""Small bounded regressors scale class-gradient templates at the current state."""
import torch
from torch import nn
from factor_surrogate import FactorSurrogate

class FactorScale(nn.Module):
    def __init__(self,dim,rank_u=128,rank_v=256,q=64,class_tokens=None,kind='factor_scale'):
        super().__init__();self.cfg=dict(dim=dim,rank_u=rank_u,rank_v=rank_v,q=q,class_tokens=class_tokens,kind=kind);self.input_field='h';self.feature_mode='hx'
        self.register_buffer('target_scale',torch.tensor(1.));self.register_buffer('U',torch.zeros(dim,rank_u));self.register_buffer('V',torch.zeros(dim,rank_v));self.register_buffer('class_tokens',torch.tensor(class_tokens));self.register_buffer('class_mean',torch.zeros(3,rank_u,rank_v))
        self.register_buffer('feature_mean',torch.zeros(6*dim+4));self.register_buffer('feature_basis',torch.zeros(6*dim+4,q));self.register_buffer('feature_std',torch.ones(q));self.register_buffer('regression',torch.zeros(3,q+2,2))

    def design(self,features):
        z=(features-self.feature_mean)@self.feature_basis/self.feature_std
        return torch.cat([z,features[:,-1:],torch.ones(len(z),1,device=z.device)],1)

    def scales(self,features,classes):
        x=self.design(features);return (1+torch.einsum('bd,bdf->bf',x,self.regression[classes])).clamp(0,3)

    def predict_factors(self,d,A,B,s):
        features,classes=FactorSurrogate.features(self,d);C=self.class_mean[classes];scale=self.scales(features,classes)
        ga=s*((B.T@self.U)[None]@C@self.V.T);gb=s*(self.U[None]@C@(self.V.T@A.T))
        return ga*scale[:,0,None,None],gb*scale[:,1,None,None]
