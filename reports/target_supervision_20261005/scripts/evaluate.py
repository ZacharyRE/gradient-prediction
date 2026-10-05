from common import *
import argparse, sys

GRADING_REVISION='v3_unambiguous_choice_and_bool_before_final'

def last_box(s):
 pos=s.rfind('\\boxed{')
 if pos<0:return None
 depth=1;j=pos+7
 while j<len(s) and depth:depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[pos:j] if depth==0 else None

def grade_legacy(text,r):
 if r.get('answer_type') in ['choice','bool']:
  box=last_box(text)
  value=box[7:-1].strip() if box else text.strip().split('\n')[-1].strip()
  value=re.sub(r'^(?:final\s+)?answer\s*[:：]\s*','',value,flags=re.I).strip().strip('$. ')
  if r['answer_type']=='choice':
   value=value.upper();valid=re.fullmatch(r'[A-E]',value) is not None
   gold=r['gold'].upper()
  else:
   value=value.lower();valid=value in ['yes','no'];gold=r['gold'].lower()
  correct=bool(valid and value==gold)
  return dict(correct=correct,strict_correct=correct and box is not None,has_box=box is not None,extracted_answer=value)
 if r['task']=='countdown':
  sys.path.insert(0,str(ROOT/'research/0.5B_countdown_lora/scripts'))
  from countdown import check
  v=check(text,r['nums'],r['target']);return dict(correct=v['correct'],strict_correct=v['correct'] and v['has_box'],**{k:v[k] for k in ['has_box','expression','inventory_valid']})
 from math_verify import parse,verify
 gold=parse('\\boxed{'+r['gold']+'}' if 'gold' in r else r['solution'])
 box=last_box(text)
 regular=bool(gold and verify(gold,parse(text,parsing_timeout=2),timeout_seconds=2))
 strict=bool(box and gold and verify(gold,parse(box,parsing_timeout=2),timeout_seconds=2))
 return dict(correct=regular,strict_correct=strict,has_box=box is not None)

def grade(text,r):
 result=grade_legacy(text,r);result['legacy_correct']=result['correct'];result['literal_option_text']=False
 kind=r.get('answer_type')
 if kind not in ['choice','bool']:return result
 def normalize(value):
  value=value.strip()
  while len(value)>1 and value.startswith('$') and value.endswith('$'):value=value[1:-1].strip()
  wrapper=re.fullmatch(r'\\(?:text|mathrm|textbf)\{([^{}]*)\}',value)
  if wrapper:value=wrapper[1].strip()
  return re.sub(r'\s+',' ',value).casefold().removesuffix('.')
 box=last_box(text);content=box[7:-1] if box else text.strip().split('\n')[-1].strip()
 content=re.sub(r'^(?:final\s+)?answer\s*[:：]\s*','',content,flags=re.I)
 value=normalize(content)
 if kind=='bool':
  aliases={'yes':'yes','no':'no','true':'yes','false':'no'}
  if value in aliases:result.update(correct=aliases[value]==r['gold'].lower(),extracted_answer=aliases[value])
  return result
 raw_source=r.get('raw_source',{});choices=raw_source.get('choices',{})
 options=choices.get('text',[]) if isinstance(choices,dict) else []
 if not options and isinstance(raw_source.get('options'),list):
  options=[re.sub(r'^[A-E]\)\s*','',s).strip() for s in raw_source['options']]
 if box and re.fullmatch('[a-e]',value):
  result.update(correct=value.upper()==r['gold'].upper(),extracted_answer=value.upper());return result
 # No free-form letter search: a boxed literal option must have a unique match.
 if box:
  matches=[i for i,option in enumerate(options) if normalize(option)==value]
  if len(matches)==1:
   label=chr(65+matches[0]);result.update(correct=label==r['gold'].upper(),extracted_answer=label,literal_option_text=True);return result
 # Outside a box, require the whole output to be one letter plus its exact text.
 match=re.fullmatch(r'\s*([A-E])\)\s+(.+?)\s*',content if box else text,re.S)
 if match is not None:
  index=ord(match[1])-ord('A')
  if index<len(options) and normalize(match[2])==normalize(options[index]):result.update(correct=match[1]==r['gold'].upper(),extracted_answer=match[1],literal_option_text=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True)
 ap.add_argument('--tasks',nargs='+',default=['countdown','gsm8k','math']);ap.add_argument('--split',default='accuracy_dev')
 ap.add_argument('--adapters',nargs='*',default=[]);ap.add_argument('--base',action='store_true');ap.add_argument('--name',required=True)
 ap.add_argument('--max-new-tokens',type=int,default=2048);ap.add_argument('--max-seqs',type=int,default=128);ap.add_argument('--entries');ap.add_argument('--memory-fraction',type=float,default=.25);ap.add_argument('--kv-gib',type=int,default=16);ap.add_argument('--request-chunk-size',type=int,default=256)
 a=ap.parse_args();gpu_guard();guard_space();os.environ['VLLM_BATCH_INVARIANT']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 entries=json.loads(Path(a.entries).read_text()) if a.entries else None
 if entries:
  a.tasks=sorted(set(e['task'] for e in entries));a.adapters=list(dict.fromkeys(e['adapter'] for e in entries if e.get('adapter')))
 from transformers import AutoTokenizer,AutoConfig
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 tok=AutoTokenizer.from_pretrained(MODELS[a.model],local_files_only=True)
 rows={task:read(data(task,a.split)) for task in a.tasks};prompts={task:[prompt(tok,r) for r in rr] for task,rr in rows.items()}
 lengths=[len(tok.encode(s,add_special_tokens=False)) for pp in prompts.values() for s in pp]
 context=((max(lengths)+a.max_new_tokens+255)//256)*256
 assert context<=AutoConfig.from_pretrained(MODELS[a.model]).max_position_embeddings,'Refuse to truncate full inputs'
 out=EXP/'results/eval'/a.name;out.mkdir(parents=True,exist_ok=True)
 protocol_args=vars(a).copy()
 assert a.request_chunk_size>0
 if a.request_chunk_size==256:protocol_args.pop('request_chunk_size')
 if a.kv_gib==16:protocol_args.pop('kv_gib') # Preserve compatibility with the original16GiB protocol.
 protocol=dict(args=protocol_args,base=MODELS[a.model],input_sha256={t:sha(data(t,a.split)) for t in a.tasks},
  adapter_sha256={p:sha(Path(p)/'adapter_model.safetensors') for p in a.adapters},full_inputs=True,max_prompt_tokens=max(lengths),max_new_tokens=a.max_new_tokens,
  context=context,backend='native vLLM BF16 LoRA',batch_invariant=True,seed=42,temperature=0,system=SYSTEM,user=USER,gpu_memory_utilization=a.memory_fraction,kv_cache_memory_bytes=a.kv_gib*1024**3,grading_revision=GRADING_REVISION)
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol,'Cannot resume different protocol'
 else:write(out/'protocol.json',protocol)
 variants=([None] if a.base else [])+a.adapters
 sums=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
 def tag(p):return 'base' if p is None else Path(p).parent.name+'__'+Path(p).name
 jobs=entries if entries is not None else [dict(adapter=p,task=t,tag=tag(p)) for p in variants for t in a.tasks]
 if all(e['tag']+'__'+e['task'] in sums for e in jobs):return
 llm=LLM(model=MODELS[a.model],dtype='bfloat16',max_model_len=context,max_num_seqs=a.max_seqs,
  gpu_memory_utilization=a.memory_fraction,kv_cache_memory_bytes=a.kv_gib*1024**3,seed=42,enable_lora=bool(a.adapters),max_lora_rank=64,max_loras=1,
  enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'})
 try:
  for j,job in enumerate(jobs):
   adapter=job.get('adapter');task=job['task'];req=None if adapter is None else LoRARequest(job['tag'],j+1,str(Path(adapter).resolve()))
   for task in [task]:
    name=job['tag']+'__'+task
    if name in sums:continue
    path=out/f'{name}.jsonl';preds=read(path) if path.exists() else []
    assert all(p['sample_id']==r['sample_id'] for p,r in zip(preds,rows[task]))
    start=time.time()
    for st in range(len(preds),len(rows[task]),a.request_chunk_size):
     if deadline():write(out/'stopped.json',dict(reason='authorized user wall-clock deadline'));return
     chunk=rows[task][st:st+a.request_chunk_size];pp=prompts[task][st:st+a.request_chunk_size]
     outputs=llm.generate(pp,SamplingParams(temperature=0,max_tokens=a.max_new_tokens,seed=42),lora_request=req,use_tqdm=True)
     assert len(outputs)==len(chunk)
     for r,pr,o in zip(chunk,pp,outputs):
      g=o.outputs[0];preds.append(dict(sample_id=r['sample_id'],problem=r['problem'],full_prompt=pr,gold=r.get('gold',r['solution']),prediction=g.text,tokens=len(g.token_ids),finish_reason=g.finish_reason,overlap_with_experiment_training=r.get('overlap_with_experiment_training',False),**grade(g.text,r)))
     jsonl(path,preds)
    n=len(preds);sums[name]=dict(n=n,correct=sum(p['correct'] for p in preds),accuracy=sum(p['correct'] for p in preds)/n,
     strict_accuracy=sum(p['strict_correct'] for p in preds)/n,legacy_correct=sum(p['legacy_correct'] for p in preds),legacy_accuracy=sum(p['legacy_correct'] for p in preds)/n,literal_option_text_count=sum(p['literal_option_text'] for p in preds),limit_hits=sum(p['finish_reason']=='length' for p in preds),mean_tokens=sum(p['tokens'] for p in preds)/n,seconds=time.time()-start)
    if any(p.get('overlap_with_experiment_training',False) for p in preds):
     clean=[p for p in preds if not p.get('overlap_with_experiment_training',False)]
     sums[name]['overlap_free']=dict(n=len(clean),correct=sum(p['correct'] for p in clean),accuracy=sum(p['correct'] for p in clean)/len(clean),strict_accuracy=sum(p['strict_correct'] for p in clean)/len(clean))
    write(out/'summary.json',sums);log(dict(evaluation=name,**sums[name]))
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)

if __name__=='__main__':main()
