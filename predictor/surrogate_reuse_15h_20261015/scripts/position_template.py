"""Class/position activation-gradient templates retain each example's exact X."""
import torch
from torch import nn
from predictor_core import raw

class PositionTemplate(nn.Module):
    def __init__(self,dim,class_tokens,bins=4,slots=6,kind='position_template'):
        super().__init__();self.cfg=dict(dim=dim,class_tokens=class_tokens,bins=bins,slots=slots,kind=kind);self.input_field='x';self.requires_current_output=False;self.bins=bins;self.slots=slots
        self.register_buffer('target_scale',torch.tensor(1.));self.register_buffer('class_tokens',torch.tensor(class_tokens));self.register_buffer('templates',torch.zeros(3,bins+slots,dim))

    def compact(self,d,include_targets=False):
        x=d['x'].float();n,L,D=x.shape;positions=torch.arange(L,device=x.device)[None];first=d['active'].long().argmax(1);class_mask=d['label_ids'][...,None]==self.class_tokens[None,None];assert torch.all(class_mask.any(-1).sum(1)==1);classes=class_mask.long().sum(1).argmax(1)
        xx=[];gg=[]
        if self.bins:
            bin_index=(positions*self.bins//first.clamp_min(1)[:,None]).clamp_max(self.bins-1)
            for b in range(self.bins):
                mask=(positions<first[:,None])&(bin_index==b)&d['valid'];count=mask.sum(1).clamp_min(1)[:,None];xx.append((x*mask[...,None]).sum(1)/count)
                if include_targets:gg.append((d['g']*mask[...,None]).sum(1))
        slots=d['active'].long().cumsum(1)-1;assert torch.all(d['active'].sum(1)==self.slots)
        for b in range(self.slots):
            mask=d['active']&(slots==b);xx.append((x*mask[...,None]).sum(1))
            if include_targets:gg.append((d['g']*mask[...,None]).sum(1))
        return torch.stack(xx,1),classes,torch.stack(gg,1) if include_targets else None

    def predict_factors(self,d,A,B,s):
        x,classes,_=self.compact(d);return raw(x,self.templates[classes],A,B,s)
