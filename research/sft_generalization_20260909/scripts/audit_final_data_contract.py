"""CPU-only final prompt/gold contract, without any model predictions."""
import logging,re
from transformers import AutoTokenizer
from common import *
from evaluation.evaluate_sft_math_benchmarks import make_prompt,is_correct

tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
warnings=[];current={}
class Capture(logging.Handler):
 def emit(self,record):
  if record.name.startswith('math_verify'):warnings.append(dict(current,message=record.getMessage()))
handler=Capture(level=logging.WARNING);logging.getLogger().addHandler(handler)
result={}
for dataset in ['math_reused5000','ood_minerva','ood_olympiad','ood_svamp','ood_amc23']:
 rows=read(S/f'data/{dataset}.jsonl');lengths=[];bad=[];integer_controls=0;false_controls=[]
 for i,row in enumerate(rows):
  prompt=make_prompt(tok,row['problem']);lengths.append(len(tok(prompt,add_special_tokens=False)['input_ids']))
  if dataset.startswith('ood_'):
   current.clear();current.update(dataset=dataset,index=i,phase='gold_self_score')
   if not is_correct('math',row,row['solution']):bad.append(i)
   answer=row.get('answer','').strip()
   if re.fullmatch(r'-?\d+',answer):
    integer_controls+=1;current['phase']='integer_gold_plus_one_negative_control'
    if is_correct('math',row,'\\boxed{'+str(int(answer)+1)+'}'):false_controls.append(i)
 result[dataset]=dict(n=len(rows),dataset_sha256=sha(S/f'data/{dataset}.jsonl'),max_prompt_tokens=max(lengths),prompt_indices_over_context_budget=[i for i,n in enumerate(lengths) if n+2048>4096],gold_self_failures=bad,integer_negative_controls=integer_controls,integer_false_accepts=false_controls)
logging.getLogger().removeHandler(handler)
passed=all(not r['prompt_indices_over_context_budget'] and not r['gold_self_failures'] and not r['integer_false_accepts'] for r in result.values())
write(S/'audits/final_data_contract.json',dict(time=time.time(),passed=passed,datasets=result,warnings=warnings,script_sha256=sha(__file__),scope='No model output or OOD score. Check actual prompt length and unchanged scorer gold self-consistency; numeric perturbation is a limited negative control. Not a proof that public benchmark answers are mathematically correct or contamination-free. FullMATH gold checks rely on previous evaluator use; new OOD is checked here.'))
print(json.dumps(result,indent=2));assert passed
