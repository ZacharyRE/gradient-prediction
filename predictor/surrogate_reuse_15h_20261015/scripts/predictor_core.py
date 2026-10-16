from common import *
import torch
from torch import nn
from torch.nn import functional as F

def reserve_model_slot(model_key):
 gpu_guard();guard_space()
 import fcntl
 path=EXP/'results/model.lock';handle=path.open('a+')
 fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 return handle


def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=True
 # Reset process-global predictor attention mode between independent jobs.
 # LossCritic explicitly disables it again for higher derivatives.
 torch.backends.mha.set_fastpath_enabled(True)

def setup(model_key,adapter=None,rng=101,layer=8):
 slot=reserve_model_slot(model_key)
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import LoraConfig,PeftModel,get_peft_model
 seed(rng);tok=AutoTokenizer.from_pretrained(MODELS[model_key],local_files_only=True)
 m=AutoModelForCausalLM.from_pretrained(MODELS[model_key],dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda()
 if adapter:m=PeftModel.from_pretrained(m,str(adapter),is_trainable=True)
 else:m=get_peft_model(m,LoraConfig(r=64,lora_alpha=64,lora_dropout=0,target_modules=[f'model.layers.{layer}.self_attn.o_proj'],task_type='CAUSAL_LM',bias='none'))
 m.config.use_cache=False;m.eval()
 mods=[(n,x) for n,x in m.named_modules() if hasattr(x,'lora_A')];assert len(mods)==1 and f'layers.{layer}.' in mods[0][0]
 assert all(('lora_' in n)==p.requires_grad for n,p in m.named_parameters())
 import weakref
 weakref.finalize(m,slot.close)
 reserve_worker_cache()
 return m,tok,mods[0][1]

def weights(mod):return mod.lora_A['default'].weight,mod.lora_B['default'].weight,mod.scaling['default']
def versions(m):return {n:p._version for n,p in m.named_parameters() if not p.requires_grad}

class Dataset:
 def __init__(self,task,split,tok):
  self.rows=read(data(task,split));self.items=[]
  from transformers import AutoConfig
  limit=AutoConfig.from_pretrained(tok.name_or_path).max_position_embeddings
  for i,r in enumerate(self.rows):
   ids=tok.encode(prompt(tok,r),add_special_tokens=False)
   ans=tok.encode(r['solution'],add_special_tokens=False)+[tok.eos_token_id]
   if len(ids)+len(ans)>limit:raise ValueError(f'Full sequence exceeds native context: {task}/{split}/{i}, {len(ids)+len(ans)}')
   self.items.append(dict(input_ids=ids+ans,labels=[-100]*len(ids)+ans,n=len(ans),index=i))
  self.lengths=[len(r['input_ids']) for r in self.items]
 def __len__(self):return len(self.items)
 def __getitem__(self,i):return self.items[i]

def batch(ds,ids,tok):
 rr=[ds[i] for i in ids];length=((max(len(r['input_ids']) for r in rr)+7)//8)*8
 return {k:torch.tensor(v,device='cuda') for k,v in dict(
  input_ids=[r['input_ids']+[tok.pad_token_id]*(length-len(r['input_ids'])) for r in rr],
  labels=[r['labels']+[-100]*(length-len(r['labels'])) for r in rr],
  attention_mask=[[1]*len(r['input_ids'])+[0]*(length-len(r['input_ids'])) for r in rr]).items()}

def chunks(ds,ids,max_n=4,token_budget=2048):
 group=[];longest=0
 for i in ids:
  n=len(ds[i]['input_ids'])
  if group and (len(group)>=max_n or max(longest,n)*(len(group)+1)>token_budget):yield group;group=[];longest=0
  group.append(int(i));longest=max(longest,n)
 if group:yield group

def forward(m,b):
 # Match the stable arithmetic backend already audited by the repository's SFT work.
 from torch.nn.attention import sdpa_kernel,SDPBackend
 with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):return m(**b,use_cache=False)

class Capture:
 def __init__(self,mod,model=None):
  def hook(_,args,out):
   self.x=args[0].detach();self.y=out
   if out.requires_grad:out.retain_grad()
  self.handle=mod.register_forward_hook(hook)
  self.residual_norm=None;self.residual_handle=None
  if model is not None:
   self.residual_norm=model.base_model.model.model.layers[8].post_attention_layernorm
   def capture_h(_,args):self.h=args[0].detach()
   self.residual_handle=self.residual_norm.register_forward_pre_hook(capture_h)
 def close(self):
  self.handle.remove()
  if self.residual_handle is not None:self.residual_handle.remove()

def raw(x,g,A,B,s=1):
 x=x.float();g=g.float()
 return s*(g@B.float()).transpose(-1,-2)@x,s*g.transpose(-1,-2)@(x@A.float().transpose(-1,-2))

def flat(pair):return torch.cat([x.reshape(-1) for x in pair])
def measures(p,t):
 p=p.reshape(-1).float();t=t.reshape(-1).float()
 return dict(cosine=float(F.cosine_similarity(p,t,dim=0)),relative_l2=float((p-t).norm()/t.norm().clamp_min(1e-12)),norm_ratio=float(p.norm()/t.norm().clamp_min(1e-12)))

def collect(m,tok,mod,cap,ds,ids,micro=4,keep=True,verify_reconstruction=True):
 A,B,s=weights(mod);truth=[torch.zeros_like(A),torch.zeros_like(B)];samples=[];audit=[];start=time.time()
 for ix in chunks(ds,ids,max_n=micro):
  bb=batch(ds,ix,tok);m.zero_grad(set_to_none=True);n=sum(ds[i]['n'] for i in ix)
  loss=forward(m,bb).loss
  if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite CE during collection, rows={ix}')
  (loss*n).backward()
  if not all(torch.isfinite(v.grad).all() for v in [A,B]):raise FloatingPointError(f'Nonfinite LoRA gradient during collection, rows={ix}')
  truth[0]+=A.grad;truth[1]+=B.grad
  if keep:
   x=cap.x.detach();y=cap.y.detach();g=cap.y.grad.detach()
   if verify_reconstruction:
    with torch.no_grad():ga,gb=raw(x,g,A,B,s)
    audit.append(measures(flat((ga.sum(0),gb.sum(0))),flat((A.grad,B.grad))))
   for j,i in enumerate(ix):
    length=len(ds[i]['input_ids']);active=torch.tensor(ds[i]['labels'][1:]+[-100])>=0
    samples.append(dict(index=i,x=x[j,:length].cpu().bfloat16(),y=y[j,:length].cpu().bfloat16(),g=g[j,:length].cpu().float(),active=active,n=ds[i]['n'],label_ids=torch.tensor(ds[i]['labels'][1:]+[-100])))
    if cap.residual_norm is not None:samples[-1]['h']=cap.h[j,:length].cpu().bfloat16()
  del loss,bb
 m.zero_grad(set_to_none=True)
 return samples,truth,dict(seconds=time.time()-start,max_reconstruction_relative_l2=max((r['relative_l2'] for r in audit),default=0),min_reconstruction_cosine=min((r['cosine'] for r in audit),default=1))

@torch.no_grad()
def features(m,tok,mod,cap,ds,ids,micro=4):
 samples=[]
 # Prefix-only forward: stop at o_proj after its hook captured current X/Y.
 class PrefixDone(Exception):pass
 def stop(*args):raise PrefixDone()
 handle=mod.register_forward_hook(stop) if cap.residual_norm is None else cap.residual_norm.register_forward_pre_hook(stop)
 try:
  for ix in chunks(ds,ids,max_n=micro):
   bb=batch(ds,ix,tok)
   try:forward(m,bb)
   except PrefixDone:pass
   for j,i in enumerate(ix):
    length=len(ds[i]['input_ids']);active=torch.tensor(ds[i]['labels'][1:]+[-100])>=0
    samples.append(dict(index=i,x=cap.x[j,:length].cpu().bfloat16(),y=cap.y[j,:length].cpu().bfloat16(),active=active,n=ds[i]['n'],label_ids=torch.tensor(ds[i]['labels'][1:]+[-100])))
    if cap.residual_norm is not None:samples[-1]['h']=cap.h[j,:length].cpu().bfloat16()
 finally:handle.remove()
 return samples

@torch.no_grad()
def ce(m,tok,ds,ids=None,micro=4):
 ids=list(range(len(ds))) if ids is None else ids;total=count=0;per=[]
 for ix in chunks(ds,ids,max_n=micro):
  bb=batch(ds,ix,tok);n=sum(ds[i]['n'] for i in ix);loss=float(forward(m,bb).loss);total+=loss*n;count+=n
 return total/count

def oracle_step(m,tok,mod,ds,ids,opt,micro=4):
 opt.zero_grad(set_to_none=True);N=sum(ds[i]['n'] for i in ids);total=0
 for ix in chunks(ds,ids,max_n=micro):
  bb=batch(ds,ix,tok);n=sum(ds[i]['n'] for i in ix);loss=forward(m,bb).loss;(loss*n/N).backward();total+=float(loss.detach())*n
 norm=float(nn.utils.clip_grad_norm_(weights(mod)[:2],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
 return dict(loss=total/N,gradient_norm=norm)

class Predictor(nn.Module):
 def __init__(self,dim,width=512,depth=2,label_dim=0,label_vocab=0,input_field='y'):
  super().__init__();self.cfg=dict(dim=dim,width=width,depth=depth,label_dim=label_dim,label_vocab=label_vocab,input_field=input_field);self.register_buffer('target_scale',torch.tensor(1.));self.input_field=input_field
  self.label_dim=label_dim
  if label_dim:
   self.register_buffer('label_table',torch.zeros(label_vocab,label_dim,dtype=torch.bfloat16));self.label_proj=nn.Linear(label_dim,width,bias=False)
  self.norm=nn.LayerNorm(dim);self.proj=nn.Linear(dim,width);self.extra=nn.Linear(5,width,bias=False)
  self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,width*2,0.,'gelu',batch_first=True,norm_first=True),depth,enable_nested_tensor=False)
  self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dim))
 def forward(self,d):
  y=d[self.input_field].float();pos=d['pos'];rms=y.square().mean(-1).clamp_min(1e-12).sqrt().log()
  h=self.proj(self.norm(y))+self.extra(torch.stack([rms,d['active'].float(),pos,pos*pos,torch.sin(torch.pi*pos)],-1))
  if self.label_dim:
   labels=d['label_ids'];emb=self.label_table[labels.clamp_min(0)].float()*(labels>=0)[...,None]
   h=h+self.label_proj(emb)
  return self.head(self.body(h,src_key_padding_mask=~d['valid']))*self.target_scale*d['valid'][...,None]

def pack(rows):
 n=len(rows);L=max(len(r['x']) for r in rows);di=rows[0]['x'].shape[-1];do=rows[0]['y'].shape[-1]
 d=dict(x=torch.zeros(n,L,di,device='cuda',dtype=torch.bfloat16),y=torch.zeros(n,L,do,device='cuda',dtype=torch.bfloat16),
  active=torch.zeros(n,L,device='cuda',dtype=torch.bool),valid=torch.zeros(n,L,device='cuda',dtype=torch.bool),pos=torch.zeros(n,L,device='cuda'))
 if 'g' in rows[0]:d['g']=torch.zeros(n,L,do,device='cuda')
 if 'h' in rows[0]:d['h']=torch.zeros(n,L,do,device='cuda',dtype=torch.bfloat16)
 if 'label_ids' in rows[0]:d['label_ids']=torch.full((n,L),-100,device='cuda',dtype=torch.long)
 for j,r in enumerate(rows):
  l=len(r['x'])
  for k in ['x','y','active']+(['g'] if 'g' in d else [])+(['h'] if 'h' in d else [])+(['label_ids'] if 'label_ids' in d else []):d[k][j,:l]=r[k].to('cuda')
  d['valid'][j,:l]=True;d['pos'][j,:l]=torch.arange(l,device='cuda')/max(1,l-1)
 d['counts']=torch.tensor([r['n'] for r in rows],device='cuda');return d

def sample_chunks(rows,ids=None,max_n=16,budget=8192):
 ids=range(len(rows)) if ids is None else ids;group=[];longest=0
 for i in ids:
  length=len(rows[i]['x'])
  if group and (len(group)>=max_n or max(longest,length)*(len(group)+1)>budget):yield group;group=[];longest=0
  group.append(int(i));longest=max(longest,length)
 if group:yield group

def objective(p,d,A,B,s,aggregate_weight=0):
 condition_lora_state(p,d,A,B,s)
 g=p(d);pa,pb=raw(d['x'],g,A,B,s)
 with torch.no_grad():ta,tb=raw(d['x'],d['g'],A,B,s)
 la=(pa-ta).square().sum((1,2))/ta.square().sum((1,2)).clamp_min(1e-12)
 lb=(pb-tb).square().sum((1,2))/tb.square().sum((1,2)).clamp_min(1e-12)
 loss=(la.mean()+lb.mean())/2+.25*(g-d['g']).square().sum()/d['valid'].sum()/g.shape[-1]/p.target_scale.square()
 if aggregate_weight:
  pp=flat((pa.sum(0),pb.sum(0)));tt=flat((ta.sum(0),tb.sum(0)))
  loss+=aggregate_weight*(pp-tt).square().sum()/tt.square().sum().clamp_min(1e-12)
 return loss

@torch.no_grad()
def diagnose(p,rows,A,B,s,return_grads=False,factor_detail=False):
 p.eval();pp=[];tt=[];acts=[];factor=[];counts=[];factor_a=[];factor_b=[]
 for ix in sample_chunks(rows):
  d=pack([rows[i] for i in ix]);condition_lora_state(p,d,A,B,s);g=p(d);pg=raw(d['x'],g,A,B,s);tg=raw(d['x'],d['g'],A,B,s)
  for j in range(len(ix)):
   pp.append(flat((pg[0][j],pg[1][j])));tt.append(flat((tg[0][j],tg[1][j])));counts.append(rows[ix[j]]['n'])
   acts.append(measures(g[j],d['g'][j]))
   if factor_detail or return_grads:factor_a.append(measures(pg[0][j],tg[0][j]));factor_b.append(measures(pg[1][j],tg[1][j]))
   factor.append(sum(float((a[j]-b[j]).square().sum()/b[j].square().sum().clamp_min(1e-12)) for a,b in zip(pg,tg))/2)
 pp=torch.stack(pp);tt=torch.stack(tt)
 per=[measures(a,b) for a,b in zip(pp,tt)]
 batchm=measures(pp.sum(0),tt.sum(0));score=float(np.mean(factor))+batchm['relative_l2']**2
 result=dict(n=len(rows),activation={k:float(np.mean([a[k] for a in acts])) for k in acts[0]},lora={k:float(np.mean([a[k] for a in per])) for k in per[0]},batch_lora=batchm,selection_score=score,individual_score=float(np.mean(factor)))
 if factor_a:
  result['factor_A']={k:float(np.mean([r[k] for r in factor_a])) for k in factor_a[0]};result['factor_B']={k:float(np.mean([r[k] for r in factor_b])) for k in factor_b[0]}
 if return_grads:return result,pp.cpu(),tt.cpu(),counts
 return result

def save_predictor(p,path,**meta):
 torch.save(dict(config=p.cfg,state={k:v.detach().cpu() for k,v in p.state_dict().items()},**meta),path)

def condition_lora_state(p,d,A,B,s):
 if getattr(p,'requires_lora_state',False):d['lora_delta']=s*((d['x'].float()@A.float().T)@B.float().T)
 return d

def load_predictor(path):
 obj=torch.load(path,map_location='cpu',weights_only=False)
 if obj['config'].get('kind')=='loss_critic':
  from loss_critic import LossCritic
  p=LossCritic(**obj['config']).cuda()
 elif obj['config'].get('kind')=='token_critic':
  from token_critic import TokenCritic
  p=TokenCritic(**obj['config']).cuda()
 elif obj['config'].get('kind')=='copied_suffix':
  from copied_suffix import CopiedSuffix
  p=CopiedSuffix(**obj['config']).cuda()
 elif obj['config'].get('kind')=='factor_surrogate':
  from factor_surrogate import FactorSurrogate
  p=FactorSurrogate(**obj['config']).cuda()
 elif obj['config'].get('kind')=='factor_scale':
  from factor_scale import FactorScale
  p=FactorScale(**obj['config']).cuda()
 elif obj['config'].get('kind')=='position_template':
  from position_template import PositionTemplate
  p=PositionTemplate(**obj['config']).cuda()
 elif obj['config'].get('kind')=='state_predictor':
  from state_predictor import StatePredictor
  p=StatePredictor(**obj['config']).cuda()
 else:p=Predictor(**obj['config']).cuda()
 p.load_state_dict(obj['state']);p.eval();return p,obj

def load_samples(path):
 rows=[]
 for f in sorted(Path(path).glob('shard*.pt')):rows.extend(torch.load(f,map_location='cpu',weights_only=False))
 return rows
