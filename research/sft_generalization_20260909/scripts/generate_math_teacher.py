"""Math-specialized Qwen2.5-7B teacher on original2000 questions, no supplied answers."""
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
os.environ['VLLM_BATCH_INVARIANT']='1'
from evaluation.evaluate_sft_math_benchmarks import is_correct
SYSTEM_PROMPT='Please reason step by step, and put your final answer within \\boxed{}.'
USER_TEMPLATE='{problem}'

def main():
 gpu_guard();download=json.loads((S/'audits/math_teacher_download_complete.json').read_text());model=download['path'];src=S/'data/train.jsonl';rows=read(src);out=S/'results/generation/teacher_math7';out.mkdir(parents=True,exist_ok=False)
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True);prompts=[tok.apply_chat_template([dict(role='system',content=SYSTEM_PROMPT),dict(role='user',content=USER_TEMPLATE.format(problem=r['problem']))],tokenize=False,add_generation_prompt=True) for r in rows];assert max(len(tok.encode(p,add_special_tokens=False)) for p in prompts)+2048<=4096
 options=dict(dtype='bfloat16',seed=20261007,max_model_len=4096,max_num_seqs=256,gpu_memory_utilization=.35,enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,attention_config={'backend':'TRITON_ATTN'})
 write(out/'manifest.json',dict(model=model,revision=download['revision'],input_sha256=sha(src),system_prompt=SYSTEM_PROMPT,user_template=USER_TEMPLATE,teacher_prompt_source='https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct',pinned_model_card_sha256=sha(S/'literature/qwen_math7_model_card.md'),n=1,temperature=0,max_tokens=2048,engine=options,script_sha256=sha(__file__),start=time.time(),limitation='Domain-specialized teacher changes training history and uses its official CoT system prompt; cannot isolate checkpoint alone. Student train/eval template unchanged. Published scores are not this experiment. Targets require strict answer/length/process review.'));(out/'generation_source.py').write_text(Path(__file__).read_text());llm=LLM(model=model,tokenizer=model,**options)
 try:
  for start in range(0,len(rows),128):
   gpu_guard();outputs=llm.generate(prompts[start:start+128],SamplingParams(temperature=0,max_tokens=2048,seed=20261007),use_tqdm=False)
   with (out/'predictions.jsonl').open('a') as f:
    for i,o in enumerate(outputs,start):
     c=o.outputs[0];f.write(json.dumps(dict(index=i,sample=0,problem=rows[i]['problem'],prediction=c.text,correct=is_correct('math',rows[i],c.text),finish_reason=c.finish_reason,generated_tokens=len(c.token_ids),prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest()),ensure_ascii=False)+'\n')
   print('Generated',min(start+128,len(rows)),'/',len(rows),flush=True)
  write(out/'complete.json',dict(time=time.time(),rows=len(rows),samples=1))
 finally:llm.llm_engine.engine_core.shutdown(timeout=15)
if __name__=='__main__':main()
