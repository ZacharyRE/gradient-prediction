"""Freeze/cache predictor-body features; optionally calibrate only its output head."""
import types
import predictor_core as pc
import correction4
from predictor_core import *
BASE_PACK=pc.pack
BASE_COLLECT=pc.collect
CACHE=None
CACHE_DATASET=None

def pack_cached(rows):
    d=BASE_PACK(rows)
    if 'z' in rows[0]:
        d['z']=torch.zeros(len(rows),d['x'].shape[1],rows[0]['z'].shape[-1],device='cuda')
        for j,r in enumerate(rows):d['z'][j,:len(r['z'])]=r['z'].cuda()
    return d

def forward_cached(self,d):
    if 'z' not in d:return self.__dict__['original_forward'](d)
    return self.head(d['z'])*self.target_scale*d['valid'][...,None]

def configure_predictor(p):
    p.__dict__['original_forward']=p.forward
    p.forward=types.MethodType(forward_cached,p)
    for name,parameter in p.named_parameters():parameter.requires_grad_(name.startswith('head.'))
    return p

@torch.no_grad()
def encode_rows(p,rows):
    captured={}
    def hook(module,args):captured['z']=args[0].detach()
    handle=p.head.register_forward_pre_hook(hook)
    try:
        for ix in sample_chunks(rows,max_n=32,budget=8192):
            group=[rows[i] for i in ix];d=BASE_PACK(group);p.__dict__.get('original_forward',p.forward)(d)
            z=captured['z']
            for j,i in enumerate(ix):rows[i]['z']=z[j,:len(rows[i]['x'])].clone()
    finally:handle.remove()
    return rows

def collect_cached(m,tok,mod,cap,ds,ids,**kwargs):
    rows,truth,audit=BASE_COLLECT(m,tok,mod,cap,ds,ids,**kwargs)
    if ds is CACHE_DATASET:
        for r in rows:r['z']=CACHE[r['index']]['z']
    return rows,truth,audit

def install():
    pc.pack=pack_cached;correction4.pack=pack_cached
