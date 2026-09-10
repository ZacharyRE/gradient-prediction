import argparse,time,os
from common import *
os.environ['VLLM_BATCH_INVARIANT']='1'
from gradient_geometry.sft_protocol import artifact_identity
from evaluation.evaluate_sft_math_benchmarks import evaluate_one
from transformers import AutoTokenizer
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='dev');p.add_argument('--queue',action='store_true');p.add_argument('--manifest',type=Path);p.add_argument('--base',action='store_true');a=p.parse_args();gpu_guard()
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 src=S/f'data/{a.dataset}.jsonl';rows=read(src);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 options=dict(dtype='bfloat16',seed=42,max_model_len=4096,max_num_seqs=512,gpu_memory_utilization=.30,enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'},enable_lora=True,max_loras=1,max_lora_rank=64,lora_dtype='bfloat16')
 llm=LLM(model=MODEL,tokenizer=MODEL,**options);counter=0
 def run(name,adapter):
  nonlocal counter
  out=S/'results/evaluation'/a.dataset;dest=out/name/'math_summary.json'
  if dest.exists():return
  gpu_guard();counter+=1;path=Path(adapter) if adapter else None
  protocol=dict(schema=1,model=artifact_identity(Path(MODEL)),adapter=artifact_identity(path) if path else None,dataset=a.dataset,data_sha256=sha(src),samples=len(rows),script_sha256=sha(__file__),engine=options,batch_invariant=True,max_tokens=2048,temperature=0,chunk_size=500)
  summary=evaluate_one(llm,tok,SamplingParams(temperature=0,max_tokens=2048,seed=42,repetition_penalty=1.,top_p=1.,top_k=-1),'math',rows,name,LoRARequest(name,counter,str(path.resolve())) if path else None,out,500,protocol)
  print(json.dumps(summary),flush=True)
 try:
  if a.base or a.queue:run('base',None)
  if a.manifest:
   for name,adapter in json.loads(a.manifest.read_text()).items():run(name,adapter)
  if a.queue:
   while time.time()<DEADLINE:
    if (S/'STOP_EVALUATOR').exists():break
    priority_path=S/'results/dev_priority.json'
    priority=json.loads(priority_path.read_text()) if priority_path.exists() else []
    order={name:i for i,name in enumerate(priority)};pending=[]
    for f in sorted((S/'results/training').glob('*/epoch*/ready.json')):
     item=json.loads(f.read_text())
     if not (S/'results/evaluation'/a.dataset/item['name']/'math_summary.json').exists():pending.append(item)
    pending.sort(key=lambda x:(order.get(x['name'],len(order)),x['name']))
    if pending:
     item=pending[0];run(item['name'],item['adapter'])
    else:time.sleep(10)
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)
if __name__=='__main__':main()
