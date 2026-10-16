"""Gradient predictor with explicit current LoRA contribution as state input."""
from predictor_core import Predictor,torch,nn

class StatePredictor(Predictor):
    def __init__(self,dim,width=128,depth=2,label_dim=0,label_vocab=0,input_field='h',kind='state_predictor'):
        super().__init__(dim,width,depth,label_dim,label_vocab,input_field)
        self.cfg['kind']=kind;self.requires_lora_state=True
        self.delta_norm=nn.LayerNorm(dim);self.delta_proj=nn.Linear(dim,width,bias=False);self.delta_rms=nn.Linear(1,width,bias=False)

    def forward(self,d):
        y=d[self.input_field].float();delta=d['lora_delta'].float();pos=d['pos'];rms=y.square().mean(-1).clamp_min(1e-12).sqrt().log();drms=delta.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt().log()
        h=self.proj(self.norm(y))+self.delta_proj(self.delta_norm(delta))+self.delta_rms(drms)
        h=h+self.extra(torch.stack([rms,d['active'].float(),pos,pos*pos,torch.sin(torch.pi*pos)],-1))
        if self.label_dim:
            labels=d['label_ids'];emb=self.label_table[labels.clamp_min(0)].float()*(labels>=0)[...,None];h=h+self.label_proj(emb)
        return self.head(self.body(h,src_key_padding_mask=~d['valid']))*self.target_scale*d['valid'][...,None]
