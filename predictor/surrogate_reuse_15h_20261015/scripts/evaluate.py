from common import *
import argparse, sys

GRADING_REVISION='v4_unambiguous_single_letter_closing_parenthesis'

def verify_input_token_ids(expected,observed,sample_id):
 assert observed is not None and expected==observed,(
  f'Full input token mismatch for {sample_id}: expected {len(expected)}, '
  f'observed {None if observed is None else len(observed)}; refuse altered or truncated inputs')

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
 # Accept only an entire single option label followed by a closing parenthesis.
 # This recognizes unambiguous answers such as A), never letters embedded in prose.
 punctuated=re.fullmatch(r'\s*([A-Ea-e])\)\s*',content if box else text)
 if punctuated is not None:
  label=punctuated[1].upper();result.update(correct=label==r['gold'].upper(),extracted_answer=label);return result
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

def reuse_source_predictions(args,protocol,jobs,rows,prompts,out,sums):
 """Reuse identical primary/alias predictions, independent of final task order."""
 phase={'test':'final','oracle_confirmation':'oracle_confirmation'}.get(args.split)
 if phase is None:return
 receipts=[]
 for job in jobs:
  task=job['task'];arms=exploration_arms();arm=arms.get(task)
  donors=[arm['source_task']] if arm else []
  if args.split=='test':
   root=arm['source_task'] if arm else task
   donors+=[name for name,value in arms.items() if value['source_task']==root and name!=task]
  key=job['tag']+'__'+task
  if key in sums:continue
  for source in dict.fromkeys(donors):
   directory=EXP/'results/eval'/f'{source}_{phase}'
   if not (directory/'protocol.json').exists() or not (directory/'summary.json').exists():continue
   old=json.loads((directory/'protocol.json').read_text());old_args=old['args']
   comparable=['base','full_inputs','input_token_ids_verified_at_generation','max_prompt_tokens','max_new_tokens','context','backend','batch_invariant',
               'seed','temperature','system','user','gpu_memory_utilization','kv_cache_memory_bytes','grading_revision']
   if any(old.get(k)!=protocol.get(k) for k in comparable):continue
   if old['input_sha256'].get(source)!=protocol['input_sha256'][task]:continue
   if any(old_args.get(k,default)!=protocol['args'].get(k,default) for k,default in
          [('model',None),('split',None),('max_seqs',128),('request_chunk_size',256)]):continue
   entry_path=old_args.get('entries')
   if not entry_path or not Path(entry_path).exists():continue
   adapter=job.get('adapter');wanted=None if adapter is None else protocol['adapter_sha256'][adapter]
   candidates=json.loads(Path(entry_path).read_text());old_sums=json.loads((directory/'summary.json').read_text())
   for donor in candidates:
    donor_adapter=donor.get('adapter');actual=None if donor_adapter is None else old['adapter_sha256'].get(donor_adapter)
    if donor_adapter is not None and actual is None:continue
    if donor['task']!=source or actual!=wanted:continue
    donor_key=donor['tag']+'__'+source;path=directory/f'{donor_key}.jsonl'
    if donor_key not in old_sums or not path.exists():continue
    predictions=read(path)
    valid=len(predictions)==len(rows[task])==old_sums[donor_key]['n'] and old_sums[donor_key]['correct']==sum(p['correct'] for p in predictions) and all(p['sample_id']==r['sample_id'] and p['full_prompt']==pr
        for p,r,pr in zip(predictions,rows[task],prompts[task]))
    if not valid:continue
    dest=out/f'{key}.jsonl';jsonl(dest,predictions)
    sums[key]=dict(old_sums[donor_key],seconds=0.0,reused_exact_predictions=True,
        original_generation_seconds=old_sums[donor_key].get('original_generation_seconds',old_sums[donor_key]['seconds']))
    receipts.append(dict(task=task,tag=job['tag'],source_task=source,source_tag=donor['tag'],
        source_predictions=str(path),source_prediction_sha256=sha(path),copied_prediction_sha256=sha(dest),
        source_protocol_sha256=sha(directory/'protocol.json'),n=len(predictions),
        all_full_prompts_and_sample_ids_equal=True,adapter_sha256=wanted))
    break
   if key in sums:break
 if receipts:
  receipt_path=out/'exact_prediction_reuse.json'
  prior=json.loads(receipt_path.read_text()) if receipt_path.exists() else []
  write(receipt_path,prior+receipts);write(out/'summary.json',sums)

def apply_final_runtime(args):
 path=EXP/'configs/final_evaluation_runtime.json'
 if args.split!='test' or not path.exists():return None
 cfg=json.loads(path.read_text());validation=EXP/'results/evaluation_batch_validation.json'
 validation_cfg=EXP/'configs/evaluation_batch_validation.json'
 assert cfg['approved'] and cfg['validation_sha256']==sha(validation),'Final runtime override lacks approved matching validation'
 result=json.loads(validation.read_text());tested=json.loads(validation_cfg.read_text())
 assert result['complete'] and result['passed'] and result['samples']>0
 assert result['validation_config_sha256']==sha(validation_cfg)
 assert cfg['max_seqs']==tested['new_max_seqs'] and cfg['request_chunk_size']==tested['new_request_chunk_size']
 requested=dict(max_seqs=args.max_seqs,request_chunk_size=args.request_chunk_size)
 args.max_seqs=cfg['max_seqs'];args.request_chunk_size=cfg['request_chunk_size']
 return dict(config_sha256=sha(path),validation_sha256=sha(validation),requested_cli=requested,
     effective=dict(max_seqs=args.max_seqs,request_chunk_size=args.request_chunk_size),
     scope='Final evaluations only; full inputs, sample set, output caps and greedy decoding unchanged',
     validated_outputs=result['samples'],limitation=result['limitation'])

def apply_development_runtime(args):
 path=EXP/'configs/development_grouped_runtime.json'
 if args.split!='accuracy_dev' or not path.exists() or '_predictor_' not in args.name:return None
 cfg=json.loads(path.read_text())
 if not all(t in cfg['tasks'] for t in args.tasks):return None
 validation=EXP/'results/development_grouped_validation.json'
 plan=EXP/'configs/development_grouped_validation.json'
 result=json.loads(validation.read_text());tested=json.loads(plan.read_text())
 assert cfg['approved'] and cfg['validation_sha256']==sha(validation)
 assert result['complete'] and result['passed'] and result['faster_overall']
 assert result['validation_config_sha256']==sha(plan)
 assert result['evaluate_source_sha256']==sha(Path(__file__)),'Evaluation code changed after grouped-runtime validation'
 for key in ['max_seqs','adapter_group_size','request_chunk_size']:
  assert cfg[key]==tested['new_max_seqs' if key=='max_seqs' else key]
 requested={key:getattr(args,key) for key in ['max_seqs','adapter_group_size','request_chunk_size']}
 args.max_seqs=cfg['max_seqs'];args.adapter_group_size=cfg['adapter_group_size'];args.request_chunk_size=cfg['request_chunk_size']
 return dict(config_sha256=sha(path),validation_sha256=sha(validation),requested_cli=requested,
             effective={key:getattr(args,key) for key in requested},scope='Remaining declared predictor development evaluations only; final runtime unchanged',
             validated_outputs=result['samples'],limitation=result['limitation'])

def run_grouped_jobs(llm,a,jobs,rows,prompts,prompt_ids,out,sums,LoRARequest,SamplingParams):
 """Batch distinct adapters; preserve each adapter's row order and output schema."""
 assert a.split!='test','Grouped adapters are not approved for final evaluation'
 pending=[(j,job) for j,job in enumerate(jobs) if job['tag']+'__'+job['task'] not in sums]
 for start in range(0,len(pending),a.adapter_group_size):
  group=pending[start:start+a.adapter_group_size];states=[]
  for j,job in group:
   adapter=job.get('adapter');assert adapter,'Grouped development requires explicit LoRA adapters'
   task=job['task'];name=job['tag']+'__'+task;path=out/f'{name}.jsonl'
   preds=read(path) if path.exists() else []
   assert len(preds)<=len(rows[task]) and all(p['sample_id']==r['sample_id'] and p['full_prompt']==pr for p,r,pr in zip(preds,rows[task],prompts[task]))
   states.append(dict(task=task,name=name,path=path,preds=preds,request=LoRARequest(job['tag'],j+1,str(Path(adapter).resolve()))))
  began=time.time()
  while any(len(s['preds'])<len(rows[s['task']]) for s in states):
   if deadline():write(out/'stopped.json',dict(reason='authorized user wall-clock deadline'));return
   inputs=[];requests=[];mapping=[]
   for state in states:
    task=state['task'];offset=len(state['preds'])
    for index in range(offset,min(offset+a.request_chunk_size,len(rows[task]))):
     inputs.append(prompts[task][index]);requests.append(state['request']);mapping.append((state,index))
   outputs=llm.generate(inputs,SamplingParams(temperature=0,max_tokens=a.max_new_tokens,seed=42),lora_request=requests,use_tqdm=True)
   assert len(outputs)==len(mapping)
   for (state,index),o in zip(mapping,outputs):
    task=state['task'];r=rows[task][index];pr=prompts[task][index]
    verify_input_token_ids(prompt_ids[task][index],o.prompt_token_ids,r['sample_id'])
    g=o.outputs[0];state['preds'].append(dict(sample_id=r['sample_id'],problem=r['problem'],full_prompt=pr,gold=r.get('gold',r['solution']),prediction=g.text,tokens=len(g.token_ids),finish_reason=g.finish_reason,overlap_with_experiment_training=r.get('overlap_with_experiment_training',False),**grade(g.text,r)))
   for state in states:jsonl(state['path'],state['preds'])
  elapsed=time.time()-began;names=[s['name'] for s in states]
  for state in states:
   preds=state['preds'];name=state['name'];n=len(preds)
   sums[name]=dict(n=n,correct=sum(p['correct'] for p in preds),accuracy=sum(p['correct'] for p in preds)/n,
     strict_accuracy=sum(p['strict_correct'] for p in preds)/n,legacy_correct=sum(p['legacy_correct'] for p in preds),legacy_accuracy=sum(p['legacy_correct'] for p in preds)/n,literal_option_text_count=sum(p['literal_option_text'] for p in preds),limit_hits=sum(p['finish_reason']=='length' for p in preds),mean_tokens=sum(p['tokens'] for p in preds)/n,
     seconds=elapsed/len(states),seconds_accounting='Equal allocation of shared group wall time; not individual adapter latency',group_wall_seconds=elapsed,group_members=names)
   if any(p.get('overlap_with_experiment_training',False) for p in preds):
    clean=[p for p in preds if not p.get('overlap_with_experiment_training',False)]
    sums[name]['overlap_free']=dict(n=len(clean),correct=sum(p['correct'] for p in clean),accuracy=sum(p['correct'] for p in clean)/len(clean),strict_accuracy=sum(p['strict_correct'] for p in clean)/len(clean))
   log(dict(evaluation=name,**sums[name]))
  write(out/'summary.json',sums)

def main():
 invocation_start=time.perf_counter()
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True)
 ap.add_argument('--tasks',nargs='+',default=['countdown','gsm8k','math']);ap.add_argument('--split',default='accuracy_dev')
 ap.add_argument('--adapters',nargs='*',default=[]);ap.add_argument('--base',action='store_true');ap.add_argument('--name',required=True)
 ap.add_argument('--max-new-tokens',type=int,default=2048);ap.add_argument('--max-seqs',type=int,default=128);ap.add_argument('--entries');ap.add_argument('--memory-fraction',type=float,default=.25);ap.add_argument('--kv-gib',type=int,default=16);ap.add_argument('--request-chunk-size',type=int,default=256)
 ap.add_argument('--adapter-group-size',type=int,default=1)
 ap.add_argument('--fresh-generation',action='store_true',help='Disable prediction reuse and require a fresh output directory for timing')
 a=ap.parse_args();gpu_guard();guard_space();os.environ['VLLM_BATCH_INVARIANT']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 entries=json.loads(Path(a.entries).read_text()) if a.entries else None
 if entries:
  a.tasks=sorted(set(e['task'] for e in entries));a.adapters=list(dict.fromkeys(e['adapter'] for e in entries if e.get('adapter')))
 runtime_override=apply_final_runtime(a)
 development_override=apply_development_runtime(a)
 assert a.adapter_group_size>=1
 if a.split=='test':assert a.adapter_group_size==1,'Final evaluation uses the separately validated serial-adapter protocol'
 final_freezes=None
 surrogate_freeze=None
 if a.split in ['selection_dev','selection_holdout']:
  from surrogate_freeze import ensure_surrogate_freeze
  surrogate_freeze=ensure_surrogate_freeze(a,entries)
 if a.split in ['test','oracle_confirmation']:
  from final_freeze import ensure_global_freeze
  assert entries is not None,'Confirmation/final evaluation requires declared frozen entries'
  final_freezes={task:ensure_global_freeze(task,[e for e in entries if e['task']==task]) for task in a.tasks}
 from transformers import AutoTokenizer,AutoConfig
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 tok=AutoTokenizer.from_pretrained(MODELS[a.model],local_files_only=True)
 rows={task:read(data(task,a.split)) for task in a.tasks};prompts={task:[prompt(tok,r) for r in rr] for task,rr in rows.items()}
 prompt_ids={task:[tok.encode(s,add_special_tokens=False) for s in pp] for task,pp in prompts.items()}
 lengths=[len(ids) for pp in prompt_ids.values() for ids in pp]
 context=((max(lengths)+a.max_new_tokens+255)//256)*256
 assert context<=AutoConfig.from_pretrained(MODELS[a.model]).max_position_embeddings,'Refuse to truncate full inputs'
 out=EXP/'results/eval'/a.name;out.mkdir(parents=True,exist_ok=True)
 if a.fresh_generation:assert not (out/'protocol.json').exists(),'Fresh-generation timing requires a new output name'
 protocol_args=vars(a).copy()
 if not a.fresh_generation:protocol_args.pop('fresh_generation')
 assert a.request_chunk_size>0
 if a.request_chunk_size==256:protocol_args.pop('request_chunk_size')
 if a.kv_gib==16:protocol_args.pop('kv_gib') # Preserve compatibility with the original16GiB protocol.
 if a.adapter_group_size==1:protocol_args.pop('adapter_group_size')
 protocol=dict(args=protocol_args,base=MODELS[a.model],input_sha256={t:sha(data(t,a.split)) for t in a.tasks},
  adapter_sha256={p:sha(Path(p)/'adapter_model.safetensors') for p in a.adapters},full_inputs=True,input_token_ids_verified_at_generation=True,max_prompt_tokens=max(lengths),max_new_tokens=a.max_new_tokens,
  context=context,backend='native vLLM BF16 LoRA',batch_invariant=True,seed=42,temperature=0,system=SYSTEM,user=USER,gpu_memory_utilization=a.memory_fraction,kv_cache_memory_bytes=a.kv_gib*1024**3,grading_revision=GRADING_REVISION)
 if final_freezes is not None:protocol['global_final_freeze']=final_freezes
 if surrogate_freeze is not None:protocol['surrogate_final_freeze']=surrogate_freeze
 if runtime_override is not None:protocol['validated_runtime_override']=runtime_override
 if development_override is not None:protocol['validated_development_runtime_override']=development_override
 if a.adapter_group_size>1:protocol['adapter_group_size']=a.adapter_group_size
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol,'Cannot resume different protocol'
 else:write(out/'protocol.json',protocol)
 variants=([None] if a.base else [])+a.adapters
 sums=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
 def tag(p):return 'base' if p is None else Path(p).parent.name+'__'+Path(p).name
 jobs=entries if entries is not None else [dict(adapter=p,task=t,tag=tag(p)) for p in variants for t in a.tasks]
 if not a.fresh_generation:reuse_source_predictions(a,protocol,jobs,rows,prompts,out,sums)
 if all(e['tag']+'__'+e['task'] in sums for e in jobs):
  write(out/'runtime_receipt.json',dict(invocation_wall_seconds=time.perf_counter()-invocation_start,all_predictions_already_available=True,new_generation_seconds=0.,fresh_generation=False));return
 before_engine=time.perf_counter();before_generation_keys=set(sums)
 llm=LLM(model=MODELS[a.model],dtype='bfloat16',max_model_len=context,max_num_seqs=a.max_seqs,
  worker_extension_cls='vllm_reservation.ReservationWorkerExtension',
  gpu_memory_utilization=a.memory_fraction,kv_cache_memory_bytes=a.kv_gib*1024**3,seed=42,enable_lora=bool(a.adapters),max_lora_rank=64,max_loras=a.adapter_group_size,
  enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'})
 engine_initialization_seconds=time.perf_counter()-before_engine
 try:
  # Claim the remaining memory only after vLLM has completed startup profiling
  # and KV-cache allocation. The cache belongs to its worker, not this client.
  llm.collective_rpc('benchmark_reserve_cache',timeout=30)
  if a.adapter_group_size>1:
   run_grouped_jobs(llm,a,jobs,rows,prompts,prompt_ids,out,sums,LoRARequest,SamplingParams)
   return
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
     expected_ids=prompt_ids[task][st:st+len(chunk)]
     for r,pr,ids,o in zip(chunk,pp,expected_ids,outputs):
      verify_input_token_ids(ids,o.prompt_token_ids,r['sample_id'])
      g=o.outputs[0];preds.append(dict(sample_id=r['sample_id'],problem=r['problem'],full_prompt=pr,gold=r.get('gold',r['solution']),prediction=g.text,tokens=len(g.token_ids),finish_reason=g.finish_reason,overlap_with_experiment_training=r.get('overlap_with_experiment_training',False),**grade(g.text,r)))
     jsonl(path,preds)
    n=len(preds);sums[name]=dict(n=n,correct=sum(p['correct'] for p in preds),accuracy=sum(p['correct'] for p in preds)/n,
     strict_accuracy=sum(p['strict_correct'] for p in preds)/n,legacy_correct=sum(p['legacy_correct'] for p in preds),legacy_accuracy=sum(p['legacy_correct'] for p in preds)/n,literal_option_text_count=sum(p['literal_option_text'] for p in preds),limit_hits=sum(p['finish_reason']=='length' for p in preds),mean_tokens=sum(p['tokens'] for p in preds)/n,seconds=time.time()-start)
    if any(p.get('overlap_with_experiment_training',False) for p in preds):
     clean=[p for p in preds if not p.get('overlap_with_experiment_training',False)]
     sums[name]['overlap_free']=dict(n=len(clean),correct=sum(p['correct'] for p in clean),accuracy=sum(p['correct'] for p in clean)/len(clean),strict_accuracy=sum(p['strict_correct'] for p in clean)/len(clean))
    write(out/'summary.json',sums);log(dict(evaluation=name,**sums[name]))
 finally:
  llm.llm_engine.engine_core.shutdown(timeout=15)
  write(out/'runtime_receipt.json',dict(invocation_wall_seconds=time.perf_counter()-invocation_start,pre_engine_seconds=before_engine-invocation_start,engine_initialization_seconds=engine_initialization_seconds,new_generation_and_grading_seconds=sum(v['seconds'] for k,v in sums.items() if k not in before_generation_keys),fresh_generation=a.fresh_generation,completed_entries=len(sums),requested_entries=len(jobs),scope='Evaluation invocation including model engine startup, generation, grading, output serialization and engine shutdown; Python imports before main and stage admission excluded'))

if __name__=='__main__':main()
