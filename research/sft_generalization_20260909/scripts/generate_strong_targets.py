"""A stronger, cached teacher; original questions only, no reference in prompt."""
import argparse
from common import *
os.environ['VLLM_BATCH_INVARIANT']='1'
from evaluation.evaluate_sft_math_benchmarks import SYSTEM_PROMPT,USER_TEMPLATE,is_correct
from transformers import AutoTokenizer
STRONG='/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137'
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=S/'data/train.jsonl');p.add_argument('--name',default='teacher32');a=p.parse_args();gpu_guard()
 from vllm import LLM,SamplingParams
 rows=read(a.input);out=S/'results/generation'/a.name;out.mkdir(parents=True,exist_ok=False)
 tok=AutoTokenizer.from_pretrained(STRONG,local_files_only=True)
 prompts=[tok.apply_chat_template([dict(role='system',content=SYSTEM_PROMPT),dict(role='user',content=USER_TEMPLATE.format(problem=r['problem']))],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rows]
 assert all(len(tok.encode(x,add_special_tokens=False))+2048<=4096 for x in prompts)
 options=dict(dtype='bfloat16',seed=20260916,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=.75,enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'})
 write(out/'manifest.json',dict(model=STRONG,input=str(a.input),input_sha256=sha(a.input),script_sha256=sha(__file__),system_prompt=SYSTEM_PROMPT,user_template=USER_TEMPLATE,enable_thinking=False,n=1,temperature=0,max_tokens=2048,engine=options,start=time.time(),status='Generated targets require independent filtering and process review before training'))
 (out/'generation_source.py').write_text(Path(__file__).read_text());llm=LLM(model=STRONG,tokenizer=STRONG,**options)
 try:
  for start in range(0,len(rows),128):
   gpu_guard();outputs=llm.generate(prompts[start:start+128],SamplingParams(temperature=0,max_tokens=2048,seed=20260916),use_tqdm=False)
   with (out/'predictions.jsonl').open('a') as f:
    for i,o in enumerate(outputs,start):
     c=o.outputs[0];f.write(json.dumps(dict(index=i,sample=0,problem=rows[i]['problem'],prediction=c.text,correct=is_correct('math',rows[i],c.text),finish_reason=c.finish_reason,generated_tokens=len(c.token_ids),prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest()),ensure_ascii=False)+'\n')
   print(json.dumps(dict(completed=min(start+128,len(rows)),total=len(rows),time=time.time())),flush=True)
  write(out/'complete.json',dict(time=time.time(),rows=len(rows),samples=1))
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)
if __name__=='__main__':main()
