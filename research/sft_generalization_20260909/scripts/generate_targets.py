import argparse,json,os,time
from common import *
os.environ['VLLM_BATCH_INVARIANT']='1'
from evaluation.evaluate_sft_math_benchmarks import make_prompt,is_correct
from transformers import AutoTokenizer
from math_verify import parse,verify, LatexExtractionConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['sample','teacher'],required=True);p.add_argument('--input',type=Path,default=FOLLOW/'data/train.jsonl');p.add_argument('--name');a=p.parse_args()
 gpu_guard();name=a.name or a.kind;out=S/'results/generation'/name;out.mkdir(parents=True,exist_ok=False)
 from vllm import LLM,SamplingParams
 model=MODEL if a.kind=='sample' else TEACHER;n=8 if a.kind=='sample' else 1
 rows=read(a.input);tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 options=dict(dtype='bfloat16',seed=20260909,max_model_len=4096,max_num_seqs=256,gpu_memory_utilization=.45,enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'})
 write(out/'manifest.json',dict(model=model,input=str(a.input),input_sha=sha(a.input),script_sha=sha(__file__),n=n,temperature=.7 if n>1 else 0,max_tokens=2048,engine=options,start=time.time()))
 llm=LLM(model=model,tokenizer=model,**options)
 sampling=SamplingParams(n=n,temperature=.7 if n>1 else 0,top_p=.95 if n>1 else 1.,max_tokens=2048,seed=20260909)
 try:
  for start in range(0,len(rows),64):
   gpu_guard();chunk=rows[start:start+64];outputs=llm.generate([make_prompt(tok,r['problem']) for r in chunk],sampling,use_tqdm=False)
   with (out/'predictions.jsonl').open('a') as f:
    for i,(r,o) in enumerate(zip(chunk,outputs),start):
     for j,c in enumerate(o.outputs):
      text=c.text;boxed=parse(text,extraction_config=[LatexExtractionConfig(boxed_match_priority=0,try_extract_without_anchor=False)]) if '\\boxed' in text else []
      good=bool(boxed and verify(parse(r['solution']),boxed))
      f.write(json.dumps(dict(index=i,sample=j,problem=r['problem'],prediction=text,correct=is_correct('math',r,text),boxed_correct=good,finish_reason=c.finish_reason,generated_tokens=len(c.token_ids)),ensure_ascii=False)+'\n')
   print(json.dumps(dict(completed=min(start+64,len(rows)),total=len(rows),time=time.time())),flush=True)
  write(out/'complete.json',dict(time=time.time(),rows=len(rows),samples=n))
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)
if __name__=='__main__':main()
