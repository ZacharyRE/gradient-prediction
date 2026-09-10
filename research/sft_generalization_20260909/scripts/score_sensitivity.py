"""Independent last-box audit; never changes original model outputs or scores."""
import argparse
import logging
from collections import Counter
from common import *
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='dev');p.add_argument('--prediction-root',type=Path);p.add_argument('--tag');p.add_argument('--models',nargs='*');a=p.parse_args()
 root=a.prediction_root or S/'results/evaluation'/a.dataset;tag=a.tag or a.dataset
 assert '/' not in tag and '..' not in tag
 warnings=[];current={}
 class WarningAudit(logging.Handler):
  def emit(self,record):
   if record.name.startswith('math_verify'):warnings.append(dict(current,logger=record.name,message=record.getMessage()))
 handler=WarningAudit(level=logging.WARNING);logging.getLogger().addHandler(handler)
 rows=read(S/f'data/{a.dataset}.jsonl');gold=[]
 for i,r in enumerate(rows):
  current.clear();current.update(phase='reference_parse',index=i);gold.append(parse(r['solution']))
 assert all(gold)
 result={};disagreements=[]
 for f in sorted(root.glob('*/math_predictions.jsonl')):
  if a.models and f.parent.name not in a.models:continue
  if not (f.parent/'math_summary.json').exists():continue
  rr=read(f);assert len(rr)==len(rows);strict=[];counts=Counter()
  for i,(r,g) in enumerate(zip(rr,gold)):
   assert r['index']==i
   current.clear();current.update(phase='prediction_parse_and_verify',model=f.parent.name,index=i);warning_start=len(warnings)
   boxes=boxed_contents(r['prediction']);ok=bool(boxes and verify(g,parse('\\boxed{'+boxes[-1]+'}')))
   for warning in warnings[warning_start:]:warning['final_strict_correct']=ok
   strict.append(int(ok));counts['original_correct']+=r['correct'];counts['strict_correct']+=ok
   counts['no_complete_box']+=not boxes;counts['length_finish']+=r['finish_reason']=='length'
   if ok!=r['correct']:
    counts['disagreements']+=1
    disagreements.append(dict(model=f.parent.name,index=i,problem=rows[i]['problem'],solution=rows[i]['solution'],prediction=r['prediction'],original_correct=r['correct'],strict_correct=ok,finish_reason=r['finish_reason']))
  result[f.parent.name]=dict(n=len(rr),counts=dict(counts),strict_vector=strict,predictions_sha256=sha(f))
 if a.models:assert set(result)==set(a.models)
 write(S/f'audits/scoring_{tag}.json',dict(script_sha256=sha(__file__),method='Require last complete boxed expression equivalent to reference; missing box is failure. Sensitivity analysis, not replacement selected after scores.',prediction_root=str(root),dataset_sha256=sha(S/f'data/{a.dataset}.jsonl'),models=result,parser_comparison_warnings=warnings))
 logging.getLogger().removeHandler(handler)
 jsonl(S/f'audits/scoring_disagreements_{tag}.jsonl',disagreements)
 print(json.dumps({k:v['counts'] for k,v in result.items()},indent=2))
if __name__=='__main__':main()
