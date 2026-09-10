import argparse,math,random,shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict,get_peft_model_state_dict
from safetensors.torch import load_file
from itertools import islice
from common import *
from training.train_lora_sft import CompletionDataset,Collator,IndexedTrainingCollator,accumulate_gradients,evaluate,autocast_context

def custom_backward(model,window,objective):
 counts=[int((b['labels'][:,1:]!=-100).sum()) for b in window];total=sum(counts);n=sum(len(b['input_ids']) for b in window);value=ce=0.
 for b in window:
  b={k:v.cuda() for k,v in b.items()}
  with torch.autocast('cuda',dtype=torch.bfloat16):o=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False)
  labels=b['labels'][:,1:];mask=labels!=-100
  losses=F.cross_entropy(o.logits[:,:-1,:].float().reshape(-1,o.logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').view_as(labels)
  ce+=float(losses.detach().sum())
  if objective=='example_mean':loss=(losses.sum(1)/mask.sum(1)).sum()/n
  elif objective=='dft':loss=(losses*torch.exp(-losses).detach()*mask).sum()/total
  else:raise ValueError(objective)
  assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
 return dict(train_token_loss=ce/total,train_objective_loss=value,supervised_tokens=total,microbatch_supervised_tokens=counts)

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--train-file',type=Path,required=True)
 p.add_argument('--seed',type=int,default=43);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--batch',type=int,default=16)
 p.add_argument('--epochs',type=int,default=8);p.add_argument('--stop',type=int,default=4);p.add_argument('--attention',choices=['eager','sdpa'],default='sdpa')
 p.add_argument('--objective',choices=['token_mean','example_mean','dft'],default='token_mean');p.add_argument('--modules',choices=['qkvo','all'],default='qkvo');p.add_argument('--rank',type=int,default=16);p.add_argument('--alpha',type=int,default=16)
 p.add_argument('--initial-attention',type=Path);p.add_argument('--probe-files',nargs='*',default=[str(S/f'data/probe_{x}.jsonl') for x in ['raw','sample','teacher']]);p.add_argument('--max-length',type=int,default=2048);p.add_argument('--resume-from',type=Path);a=p.parse_args();gpu_guard();assert a.batch%4==0
 out=S/'results/training'/a.name;out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'train_source.py');shutil.copy2(S/'scripts/common.py',out/'common_source.py')
 torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 if a.attention=='sdpa':
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);ds=CompletionDataset(a.train_file,tok,a.max_length);assert ds.truncated==0
 dev=CompletionDataset(S/'data/dev_ce.jsonl',tok,a.max_length);collator=Collator(tok.pad_token_id);generator=torch.Generator().manual_seed(a.seed)
 loader=DataLoader(range(len(ds)),batch_size=4,shuffle=True,collate_fn=IndexedTrainingCollator(ds,collator),generator=generator)
 dl=DataLoader(dev,batch_size=4,collate_fn=collator,generator=torch.Generator().manual_seed(0))
 probes={Path(f).stem:DataLoader(CompletionDataset(Path(f),tok,a.max_length),batch_size=4,collate_fn=collator,generator=torch.Generator().manual_seed(0)) for f in a.probe_files}
 model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,attn_implementation=a.attention,local_files_only=True).cuda();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 targets=['q_proj','k_proj','v_proj','o_proj']+(['gate_proj','up_proj','down_proj'] if a.modules=='all' else [])
 torch.manual_seed(a.seed);model=get_peft_model(model,LoraConfig(r=a.rank,lora_alpha=a.alpha,lora_dropout=0.,bias='none',target_modules=targets,task_type='CAUSAL_LM'))
 if a.initial_attention:
  initial=load_file(str(a.initial_attention/'adapter_model.safetensors'));set_peft_model_state_dict(model,initial);current=get_peft_model_state_dict(model)
  assert all(torch.equal(current[k].detach().cpu(),v) for k,v in initial.items())
  assert all(torch.count_nonzero(v)==0 for k,v in current.items() if '.lora_B.' in k)
  write(out/'initial_attention_match.json',dict(source=str(a.initial_attention),sha256=sha(a.initial_attention/'adapter_model.safetensors'),all_equal=True,n_tensors=len(initial)))
 for param in model.parameters():
  if not param.requires_grad:param.data=param.data.to(torch.bfloat16)
 params=[v for v in model.parameters() if v.requires_grad];assert all(v.dtype==torch.float32 for v in params)
 model.save_pretrained(out/'initial_adapter');optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01,fused=True)
 manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},model=MODEL,data_sha=sha(a.train_file),script_sha=sha(__file__),examples=len(ds),trainable=sum(v.numel() for v in params),start=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'])
 manifest['probe_sha256']={f:sha(f) for f in a.probe_files}
 start_epoch=1;horizon=len(ds)*a.epochs
 if a.resume_from:
  source_manifest=json.loads((a.resume_from/'manifest.json').read_text());state=torch.load(a.resume_from/'resume_latest.pt',map_location='cpu',weights_only=False)
  assert source_manifest['data_sha']==manifest['data_sha']
  for key in ['seed','lr','batch','epochs','attention','objective','modules','rank','alpha','max_length']:
   assert state['arguments'][key]==getattr(a,key),(key,state['arguments'][key],getattr(a,key))
  assert int(state['epoch'])<min(a.stop,a.epochs)
  adapter=Path(state['adapter']);assert adapter.resolve()==(a.resume_from/f"epoch{state['epoch']}").resolve()
  saved=load_file(str(adapter/'adapter_model.safetensors'));set_peft_model_state_dict(model,saved);loaded=get_peft_model_state_dict(model)
  assert set(saved)==set(loaded) and all(torch.equal(v,loaded[k].detach().cpu()) for k,v in saved.items())
  optimizer.load_state_dict(state['optimizer']);generator.set_state(state['generator']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
  step,seen,tokens=int(state['step']),int(state['seen']),int(state['tokens']);start_epoch=int(state['epoch'])+1
  history=json.loads((a.resume_from/'history.json').read_text());exposure=json.loads((a.resume_from/'exposure.json').read_text())
  assert history[-1]['step']==step and history[-1]['seen']==seen and len(exposure)==step
  assert all(v.dtype==torch.float32 for p in params for k,v in optimizer.state[p].items() if k in ['exp_avg','exp_avg_sq'])
  manifest.update(resumed_from=str(a.resume_from),resume_adapter_sha256=sha(adapter/'adapter_model.safetensors'),resume_state_sha256=sha(a.resume_from/'resume_latest.pt'),resume_source_manifest_sha256=sha(a.resume_from/'manifest.json'),resume_start_epoch=start_epoch,resume_seen=seen,initial_adapter_is_pre_resume_constructor_only=True)
  write(out/'resume_audit.json',dict(adapter_loaded_exactly=True,optimizer_state_restored=True,shuffle_generator_exact=torch.equal(generator.get_state(),state['generator']),start_epoch=start_epoch,seen=seen,horizon=horizon,limitation='State/config checks; not a separate bitwise comparison against an uninterrupted8-epoch run. Original checkpoints preserved; new epochs written in this branch.'))
 else:
  history=[dict(step=0,seen=0,dev_ce=evaluate(model,dl,'cuda','bfloat16'),probe_ce={n:evaluate(model,l,'cuda','bfloat16') for n,l in probes.items()})];exposure=[];step=seen=tokens=0
 write(out/'manifest.json',manifest)
 for epoch in range(start_epoch,min(a.stop,a.epochs)+1):
  it=iter(loader)
  while True:
   indexed=list(islice(it,a.batch//4))
   if not indexed:break
   gpu_guard();model.train();optimizer.zero_grad(set_to_none=True);window=[b for b,_ in indexed]
   m=accumulate_gradients(model,window,'cuda','token_mean','bfloat16') if a.objective=='token_mean' else custom_backward(model,window,a.objective)
   norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
   scale=seen/64 if seen<64 else max(0.,.5*(1+math.cos(math.pi*(seen-64)/(horizon-64))))
   lr=a.lr*scale
   for g in optimizer.param_groups:g['lr']=lr
   optimizer.step();step+=1;indices=[i for _,ix in indexed for i in ix];seen+=len(indices);tokens+=m['supervised_tokens']
   h=dict(step=step,epoch=epoch,seen=seen,tokens=tokens,lr=lr,gradient_norm=float(norm),time=time.time(),**m);history.append(h);exposure.append(dict(step=step,epoch=epoch,indices=indices))
  h['dev_ce']=evaluate(model,dl,'cuda','bfloat16');h['probe_ce']={n:evaluate(model,l,'cuda','bfloat16') for n,l in probes.items()};dest=out/f'epoch{epoch}';dest.mkdir();model.save_pretrained(dest);tok.save_pretrained(dest)
  state=dict(optimizer=optimizer.state_dict(),epoch=epoch,step=step,seen=seen,tokens=tokens,generator=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),adapter=str(dest.resolve()),arguments=manifest['arguments'])
  torch.save(state,out/'resume_latest.tmp');(out/'resume_latest.tmp').replace(out/'resume_latest.pt')
  write(out/'history.json',history);write(out/'exposure.json',exposure);write(dest/'ready.json',dict(name=a.name+f'_epoch{epoch}',adapter=str(dest.resolve()),epoch=epoch,train_sha=manifest['data_sha'],time=time.time()))
  print(json.dumps(h),flush=True)
 write(out/'complete.json',dict(time=time.time(),steps=step,examples=seen,tokens=tokens,epochs=epoch))
if __name__=='__main__':main()
