"""Development-only expected single-sample accuracy, distinct from pass@N."""
import argparse,numpy as np
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
os.environ['VLLM_BATCH_INVARIANT']='1'
from evaluation.evaluate_sft_math_benchmarks import make_prompt,is_correct
from gradient_geometry.sft_protocol import artifact_identity,sample_identity
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();gpu_guard();plan=json.loads(a.plan.read_text())
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 from transformers import AutoTokenizer
 src=S/'data/dev.jsonl';assert sha(src)==plan['data_sha256'];allrows=read(src);rows=[allrows[i] for i in plan['indices']]
 out=S/'results/sampling'/plan['name'];out.mkdir(parents=True,exist_ok=False);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 options=dict(dtype='bfloat16',seed=42,max_model_len=4096,max_num_seqs=512,gpu_memory_utilization=.30,enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'},enable_lora=True,max_loras=1,max_lora_rank=64,lora_dtype='bfloat16')
 write(out/'plan.json',dict(plan,script_sha256=sha(__file__),engine=options,start=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 (out/'sampling_source.py').write_text(Path(__file__).read_text());prompts=[make_prompt(tok,r['problem']) for r in rows]
 params=[SamplingParams(n=plan['n_samples'],temperature=plan['temperature'],top_p=plan['top_p'],top_k=-1,max_tokens=2048,repetition_penalty=1.,seed=plan['decode_seed']+10*i) for i in plan['indices']]
 llm=LLM(model=MODEL,tokenizer=MODEL,**options);values={}
 try:
  for counter,(name,adapter) in enumerate(plan['models'].items(),1):
   gpu_guard();dest=out/name;dest.mkdir();path=Path(adapter) if adapter else None
   if path:
    assert sha(path/'adapter_model.safetensors')==plan['adapter_sha256'][name]
    assert artifact_identity(path)['files']==plan['adapter_files_sha256'][name]
   write(dest/'manifest.json',dict(model=artifact_identity(Path(MODEL)),adapter=artifact_identity(path) if path else None,plan_sha256=sha(a.plan),n=len(rows),n_samples=plan['n_samples'],start=time.time()))
   outputs=llm.generate(prompts,params,lora_request=LoRARequest(name,counter,str(path.resolve())) if path else None,use_tqdm=False);predictions=[]
   for j,o in enumerate(outputs):
    assert len(o.outputs)==plan['n_samples']
    samples=[dict(sample=k,prediction=c.text,correct=bool(is_correct('math',rows[j],c.text)),finish_reason=c.finish_reason,generated_tokens=len(c.token_ids)) for k,c in enumerate(o.outputs)]
    predictions.append(dict(index=plan['indices'][j],sample_hash=sample_identity(rows[j]),samples=samples,mean_single_sample_accuracy=sum(x['correct'] for x in samples)/len(samples)))
   jsonl(dest/'predictions.jsonl',predictions);v=np.array([r['mean_single_sample_accuracy'] for r in predictions]);values[name]=v
   summary=dict(n_questions=len(rows),n_samples=plan['n_samples'],mean_single_sample_accuracy=float(v.mean()),any_correct_fraction=float(np.mean(v>0)),truncated=sum(x['finish_reason']=='length' for r in predictions for x in r['samples']),time=time.time())
   write(dest/'summary.json',summary);print(name,json.dumps(summary),flush=True)
  comparisons={};base=values['base']
  for name,v in values.items():
   if name=='base':continue
   d=v-base;rng=np.random.default_rng(20260923);boots=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*100
   comparisons[name]=dict(delta_pp=float(d.mean()*100),question_ci95_pp=np.quantile(boots,[.025,.975]).tolist(),limitation='Development-only pilot; one training seed unless manifest specifies others. Fixed finite decoding streams and questions; no fresh OOD conclusion.')
  write(out/'analysis.json',dict(comparisons=comparisons,definition='Mean correctness over all samples, estimating expected accuracy of one draw. Any-correct fraction is separately labeled and is not pass@1.',time=time.time()))
  write(out/'complete.json',dict(time=time.time()))
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)
 # Optional serialized diagnostics run only after inference engine teardown.
 if plan.get('after_sampling_tasks'):
  import subprocess
  dispatched=[]
  for task in plan['after_sampling_tasks']:
   gpu_guard()
   with Path(task['log']).open('w') as log:
    result=subprocess.run(task['command'],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
   dispatched.append(dict(name=task['name'],command=task['command'],returncode=result.returncode,time=time.time()))
   write(out/'followup_dispatch.json',dispatched)
if __name__=='__main__':main()
