"""Explicit, FP32 LoRA-induced change in module output as predictor state input."""
import types
from predictor_core import *
import predictor_core as pc
import correction4

BASE_PACK=pc.pack
BASE_COLLECT=pc.collect
CONTEXT=None

@torch.no_grad()
def delta_output(x,p):
    A,B,s=p.__dict__['lora_context'];x=x.float()
    # Expand the difference to avoid subtracting nearly equal module outputs.
    da=A.detach().float()-p.warm_A.float();db=B.detach().float()-p.warm_B.float()
    return s*((x@A.detach().float().T)@db.T+(x@da.T)@p.warm_B.float().T)

def state_forward(self,d):
    y=d[self.input_field].float();pos=d['pos'];rms=y.square().mean(-1).clamp_min(1e-12).sqrt().log()
    h=self.proj(self.norm(y))+self.extra(torch.stack([rms,d['active'].float(),pos,pos*pos,torch.sin(torch.pi*pos)],-1))
    delta=d['delta_y'] if 'delta_y' in d else delta_output(d['x'],self)
    normalized=delta/(rms.exp()[...,None].clamp_min(1e-4))
    h=h+self.delta_gain*self.delta_proj(normalized)
    if self.label_dim:
        labels=d['label_ids'];emb=self.label_table[labels.clamp_min(0)].float()*(labels>=0)[...,None]
        h=h+self.label_proj(emb)
    return self.head(self.body(h,src_key_padding_mask=~d['valid']))*self.target_scale*d['valid'][...,None]

def augment(p,A,B,s,warm_A,warm_B,job):
    global CONTEXT
    p.register_buffer('warm_A',warm_A.detach().clone());p.register_buffer('warm_B',warm_B.detach().clone())
    p.__dict__['lora_context']=(A,B,s)
    p.delta_proj=nn.Linear(p.cfg['dim'],p.cfg['width'],bias=False).cuda()
    nn.init.zeros_(p.delta_proj.weight);p.delta_gain=job.get('delta_gain',10.)
    p.forward=types.MethodType(state_forward,p)
    if job.get('cal_scope') in ['delta','delta_head']:
        for name,parameter in p.named_parameters():
            parameter.requires_grad_(name.startswith('delta_proj.') or (job['cal_scope']=='delta_head' and name.startswith('head.')))
    CONTEXT=p
    return p

def state_pack(rows):
    d=BASE_PACK(rows)
    if 'delta_y' in rows[0]:
        d['delta_y']=torch.zeros_like(d['y'],dtype=torch.float32)
        for j,r in enumerate(rows):d['delta_y'][j,:len(r['x'])]=r['delta_y'].cuda()
    return d

def state_collect(*args,**kwargs):
    rows,truth,audit=BASE_COLLECT(*args,**kwargs)
    if CONTEXT is not None:
        for r in rows:r['delta_y']=delta_output(r['x'].cuda(),CONTEXT).cpu()
    return rows,truth,audit

def install():
    pc.pack=state_pack;correction4.pack=state_pack

def load_state_checkpoint(path,A,B,s):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    p=Predictor(**obj['config']).cuda()
    base={k:v for k,v in obj['state'].items() if k not in ['warm_A','warm_B'] and not k.startswith('delta_proj.')}
    p.load_state_dict(base)
    p=augment(p,A,B,s,obj['state']['warm_A'].cuda(),obj['state']['warm_B'].cuda(),obj)
    p.load_state_dict(obj['state']);p.eval();return p,obj
