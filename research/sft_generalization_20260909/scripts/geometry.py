import argparse,torch
from safetensors.torch import load_file
from common import *
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
models=json.loads(a.manifest.read_text());cache={};out={}
for name,path in models.items():
 if path is None:continue
 path=Path(path);cfg=json.loads((path/'adapter_config.json').read_text());state=load_file(str(path/'adapter_model.safetensors'));scale=cfg['lora_alpha']/cfg['r'];pairs={};per={}
 for k,v in state.items():
  if '.lora_A.' not in k:continue
  bkey=k.replace('.lora_A.','.lora_B.');A=v.double();B=state[bkey].double();key=k.split('.lora_A.')[0];pairs[key]=(A,B,scale)
  sq=float(((B.T@B)*(A@A.T).T).sum())*scale**2;per[key]=max(0,sq)**.5
 out[name]=dict(effective_delta_frobenius=sum(x*x for x in per.values())**.5,per_matrix=per,adapter_sha=sha(path/'adapter_model.safetensors'),config_sha=sha(path/'adapter_config.json'));cache[name]=pairs
cross={}
for n,x in cache.items():
 for m,y in cache.items():
  if n>=m or set(x)!=set(y):continue
  dot=0
  for key,(A,B,s) in x.items():
   C,D,t=y[key];dot+=float(((B.T@D)*(C@A.T).T).sum())*s*t
  denominator=out[n]['effective_delta_frobenius']*out[m]['effective_delta_frobenius']
  cross[n+'__'+m]=dot/denominator if denominator else None
write(a.output,dict(models=out,cosines=cross,meaning='Effective LoRA weight delta BA including alpha/r; parameter norm is not a behavioral distance'))
