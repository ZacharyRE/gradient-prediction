"""Apply prospectively corrected Minerva numeric grading; preserve legacy outputs."""
import logging
from collections import Counter
from common import *
from minerva_numeric import *
pp=S/'results/confirmation_plan.json';plan=json.loads(pp.read_text())
assert sha(__file__)==plan['minerva_numeric_scoring_script_sha256']
assert sha(S/'scripts/minerva_numeric.py')==plan['minerva_numeric_helper_sha256']
cp=S/'audits/minerva_numeric_contract.json';assert sha(cp)==plan['minerva_numeric_contract_sha256'];contract=json.loads(cp.read_text());assert contract['passed']
dataset='ood_minerva';src=S/f'data/{dataset}.jsonl';assert sha(src)==contract['dataset_sha256']==plan['data_sha256'][dataset]
root=S/'results/evaluation'/dataset;rows=read(src);names=list(json.loads((S/'results/confirmation_manifest.json').read_text()))
legacy_path=S/'audits/scoring_confirmation_ood_minerva.json';legacy=json.loads(legacy_path.read_text());assert set(legacy['models'])==set(names)
assert legacy['script_sha256']==plan['strict_scoring_script_sha256'] and legacy['dataset_sha256']==sha(src)
references={r['index']:r['numeric_value'] for r in contract['numeric_references']};assert len(references)==191
warnings=[];current={}
class Capture(logging.Handler):
 def emit(self,record):
  if record.name.startswith('math_verify'):warnings.append(dict(current,logger=record.name,message=record.getMessage()))
handler=Capture(level=logging.WARNING);logging.getLogger().addHandler(handler);result={}
try:
 for name in names:
  path=root/name/'math_predictions.jsonl';rr=read(path);assert len(rr)==272
  assert legacy['models'][name]['predictions_sha256']==sha(path)
  original=[int(r['correct']) for r in rr];oldstrict=legacy['models'][name]['strict_vector']
  primary=list(original);strict=list(oldstrict);sens={str(t):dict(primary_vector=list(original),strict_vector=list(oldstrict)) for t in contract['sensitivity_rtols']};diagnostics=[];reasons=Counter()
  for i,g in references.items():
   assert gold_numeric(rows[i]['answer'])==g
   current.clear();current.update(phase='minerva_numeric_prediction',model=name,index=i);ws=len(warnings)
   p=numeric_prediction(rr[i]['prediction']);reasons[p['reason']]+=1
   ok=bool(numeric_equal(p['value'],g,contract['primary_rtol']));primary[i]=strict[i]=int(ok)
   for w in warnings[ws:]:w['final_strict_correct']=ok
   if p['reason'] in ['numeric_evaluation_timeout','parse_exception']:
    warnings.append(dict(current,logger='minerva_numeric',message=p['reason'],final_strict_correct=ok))
   diagnostics.append(dict(index=i,gold=g,parsed_prediction=p,correct=ok))
   for t in contract['sensitivity_rtols']:
    value=int(numeric_equal(p['value'],g,t));sens[str(t)]['primary_vector'][i]=sens[str(t)]['strict_vector'][i]=value
  result[name]=dict(n=272,predictions_sha256=sha(path),counts=dict(legacy_original_correct=sum(original),legacy_strict_correct=sum(oldstrict),original_correct=sum(primary),strict_correct=sum(strict),numeric_correct=sum(primary[i] for i in references)),primary_vector=primary,strict_vector=strict,sensitivity_vectors=sens,numeric_parse_reasons=dict(reasons),numeric_cases=diagnostics)
finally:logging.getLogger().removeHandler(handler)
out=S/'audits/scoring_minerva_numeric_corrected.json'
assert not out.exists(),'Corrected score audit already exists; do not overwrite it'
write(out,dict(time=time.time(),script_sha256=sha(__file__),helper_sha256=sha(S/'scripts/minerva_numeric.py'),contract_sha256=sha(cp),plan_sha256=sha(pp),dataset_sha256=sha(src),prediction_root=str(root),legacy_strict_audit_sha256=sha(legacy_path),models=result,parser_comparison_warnings=warnings,primary_rtol=contract['primary_rtol'],primary_atol=0.,scope=contract['scope']))
print(json.dumps({n:r['counts'] for n,r in result.items()},indent=2))
