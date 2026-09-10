"""Frozen development diagnostic: equal4096 generation budget for baseline and adapters."""
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
os.environ['VLLM_BATCH_INVARIANT']='1'
from evaluation.evaluate_sft_math_benchmarks import evaluate_one
from gradient_geometry.sft_protocol import artifact_identity
from transformers import AutoTokenizer

def main():
 gpu_guard();plan_path=S/'audits/long_decode_plan.json';plan=json.loads(plan_path.read_text());src=S/'data/dev.jsonl';assert sha(src)==plan['data_sha256'];rows=read(src)
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);options=dict(dtype='bfloat16',seed=42,max_model_len=6144,max_num_seqs=512,gpu_memory_utilization=.30,enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'},enable_lora=True,max_loras=1,max_lora_rank=64,lora_dtype='bfloat16')
 llm=LLM(model=MODEL,tokenizer=MODEL,**options);out=S/'results/evaluation_long/dev';out.mkdir(parents=True,exist_ok=False);(out/'evaluation_source.py').write_text(Path(__file__).read_text())
 try:
  for i,(name,adapter) in enumerate(plan['models'].items(),1):
   gpu_guard();path=Path(adapter) if adapter else None
   if path:assert artifact_identity(path)['files']==plan['adapter_files_sha256'][name]
   protocol=dict(schema=1,model=artifact_identity(Path(MODEL)),adapter=artifact_identity(path) if path else None,dataset='dev',data_sha256=sha(src),samples=len(rows),script_sha256=sha(__file__),engine=options,batch_invariant=True,max_tokens=4096,temperature=0,chunk_size=500,plan_sha256=sha(plan_path))
   summary=evaluate_one(llm,tok,SamplingParams(temperature=0,max_tokens=4096,seed=42,repetition_penalty=1.,top_p=1.,top_k=-1),'math',rows,name,LoRARequest(name,i,str(path.resolve())) if path else None,out,500,protocol);print(json.dumps(summary),flush=True)
  write(out/'complete.json',dict(time=time.time()))
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)
if __name__=='__main__':main()
