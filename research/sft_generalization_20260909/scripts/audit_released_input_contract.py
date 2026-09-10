"""CPU audit of actual prompt functions and all currently queued released targets."""
from common import *
import argparse
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids
from evaluation.evaluate_sft_math_benchmarks import make_prompt

parser=argparse.ArgumentParser();parser.add_argument('--tag',default='released_input_contract');args=parser.parse_args()
assert args.tag.replace('_','').isalnum()
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
queue=json.loads((S/'results/train_queue.json').read_text())
files=sorted(set(j['train_file'] for j in queue));prompts={};targets={};reports=[]
special=set(tok.all_special_ids)
for file in files:
 rows=read(file);lengths=[];embedded=[]
 for row in rows:
  problem=row['problem'];solution=row['solution']
  if problem not in prompts:
   train=chat_prompt_ids(tok,problem)
   evaluation=tok(make_prompt(tok,problem),add_special_tokens=False)['input_ids']
   assert train==evaluation,('prompt_mismatch',file,row.get('original_index'))
   prompts[problem]=train
  if solution not in targets:targets[solution]=tok(solution,add_special_tokens=False)['input_ids']
  response=targets[solution];length=len(prompts[problem])+len(response)+1
  assert response and length<=2048,('length',file,length)
  unexpected=sorted(set(response)&special)
  if unexpected:embedded.append(dict(original_index=row.get('original_index'),expansion_index=row.get('expansion_index'),special_token_ids=unexpected))
  lengths.append(length)
 reports.append(dict(path=file,sha256=sha(file),n=len(rows),max_total_tokens=max(lengths),embedded_special_tokens=embedded))
assert tok.eos_token_id is not None
write(S/f'audits/{args.tag}.json',dict(time=time.time(),script_sha256=sha(__file__),queue_sha256=sha(S/'results/train_queue.json'),model=MODEL,files=reports,unique_questions=len(prompts),unique_target_texts=len(targets),all_train_eval_prompt_ids_equal=True,all_targets_fit_2048_with_eos=True,eos_token_id=tok.eos_token_id,eos_token=tok.eos_token,embedded_special_target_count=sum(len(r['embedded_special_tokens']) for r in reports),sources={str(p):sha(p) for p in [OLD/'snapshots/training/train_lora_sft.py',OLD/'snapshots/evaluation/evaluate_sft_math_benchmarks.py',OLD/'snapshots/gradient_geometry/extraction.py']},scope='Actual prompt functions and tokenizer for currently released training files only. Does not certify mathematical correctness or rerun the historical mask/causal-shift gradient audit.'))
print(json.dumps(dict(files=len(files),questions=len(prompts),target_texts=len(targets),embedded_special_target_rows=sum(len(r['embedded_special_tokens']) for r in reports))))
