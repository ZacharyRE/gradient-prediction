"""Probe known-wrong numeric answers without using any new model predictions."""
import logging,math
from sympy import Basic,latex
from math_verify import parse,verify
from common import *
from minerva_numeric import numeric_prediction
assert not (S/'results/confirmation_plan.json').exists()
assert not list((S/'results/evaluation/math_reused5000').glob('*/math_predictions.jsonl'))
warnings=[];current={}
class Capture(logging.Handler):
 def emit(self,record):
  if record.name.startswith('math_verify'):warnings.append(dict(current,message=record.getMessage()))
handler=Capture(level=logging.WARNING);logging.getLogger().addHandler(handler);results={}
try:
 for dataset in ['dev','math_reused5000']:
  rows=read(S/f'data/{dataset}.jsonl');tested=0;accepted=[];skipped=[]
  for i,r in enumerate(rows):
   current.clear();current.update(dataset=dataset,index=i,phase='gold_parse');gold=parse(r['solution']);exprs=[x for x in gold if isinstance(x,Basic)]
   if len(exprs)!=1:continue
   g=exprs[0]
   if not getattr(g,'is_number',False) or g.free_symbols or g.is_real is not True:continue
   # A true factor-two perturbation is certainly wrong for nonzero real scalars.
   if g.is_zero is None:skipped.append(dict(index=i,reason='unknown_zero_status'));continue
   bad=g*2 if g.is_zero is False else g+1
   negative=r'\boxed{'+latex(bad)+'}'
   current['phase']='numeric_factor_two_or_zero_plus_one';ok=bool(verify(gold,parse(negative)));tested+=1
   if ok:accepted.append(dict(index=i,problem=r['problem'],solution=r['solution'],parsed_gold=str(g),known_wrong_prediction=negative,negative_accepted=True))
  results[dataset]=dict(dataset_sha256=sha(S/f'data/{dataset}.jsonl'),n=len(rows),numeric_references_tested=tested,false_accepts=accepted,skipped=skipped)
finally:logging.getLogger().removeHandler(handler)
write(S/'audits/math_numeric_negative_controls.json',dict(time=time.time(),script_sha256=sha(__file__),datasets=results,warnings=warnings,scope='Before fullMATH model outputs; only a known-wrong numeric perturbation diagnostic. Zero falseaccepts would not prove scoring universallycorrect; positivecases require report/sensitivity before final freeze. No reference or score changed.'))
print(json.dumps({d:dict(tested=r['numeric_references_tested'],false_accept_indices=[x['index'] for x in r['false_accepts']]) for d,r in results.items()}))
