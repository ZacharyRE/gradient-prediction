"""Execute the real corrected scorer on clearly synthetic positive/negative fixtures."""
import runpy,shutil
import common
from common import *
from minerva_numeric import gold_numeric
actual=S;fixture=S/'audits/minerva_synthetic_fixture';assert not fixture.exists();fixture.mkdir()
for directory in ['scripts','data','audits','results/evaluation/ood_minerva','results']:(fixture/directory).mkdir(parents=True,exist_ok=True)
for f in ['scripts/minerva_numeric.py','data/ood_minerva.jsonl','audits/minerva_numeric_contract.json']:shutil.copy2(actual/f,fixture/f)
rows=read(actual/'data/ood_minerva.jsonl');legacy={}
for name,positive in [('synthetic_correct',True),('synthetic_factor_two_wrong',False)]:
 out=fixture/'results/evaluation/ood_minerva'/name;out.mkdir();pred=[]
 for i,r in enumerate(rows):
  g=gold_numeric(r['answer']);v=g if positive or g is None else (2*g if g!=0 else 1.)
  text=r'\boxed{'+format(v,'.17g')+'}' if g is not None else r['solution']
  pred.append(dict(index=i,prediction=text,correct=True,finish_reason='stop',generated_tokens=1,sample_hash='SYNTHETIC_NOT_MODEL_OUTPUT'))
 jsonl(out/'math_predictions.jsonl',pred)
 legacy[name]=dict(strict_vector=[0]*272,predictions_sha256=sha(out/'math_predictions.jsonl'))
write(fixture/'results/confirmation_manifest.json',{name:'SYNTHETIC_NO_ADAPTER' for name in legacy})
write(fixture/'results/confirmation_plan.json',dict(minerva_numeric_scoring_script_sha256=sha(actual/'scripts/score_minerva_numeric.py'),minerva_numeric_helper_sha256=sha(actual/'scripts/minerva_numeric.py'),minerva_numeric_contract_sha256=sha(actual/'audits/minerva_numeric_contract.json'),strict_scoring_script_sha256=sha(actual/'scripts/score_sensitivity.py'),data_sha256={'ood_minerva':sha(actual/'data/ood_minerva.jsonl')},scope='SYNTHETIC_SCORER_TEST_NOT_FINAL_SELECTION'))
write(fixture/'audits/scoring_confirmation_ood_minerva.json',dict(script_sha256=sha(actual/'scripts/score_sensitivity.py'),dataset_sha256=sha(actual/'data/ood_minerva.jsonl'),models=legacy))
common.S=fixture
try:runpy.run_path(str(actual/'scripts/score_minerva_numeric.py'),run_name='__main__')
finally:common.S=actual
out=json.loads((fixture/'audits/scoring_minerva_numeric_corrected.json').read_text());good=out['models']['synthetic_correct'];bad=out['models']['synthetic_factor_two_wrong']
assert sum(good['primary_vector'])==272 and sum(good['strict_vector'])==191
assert sum(bad['primary_vector'])==81 and sum(bad['strict_vector'])==0
indices=[i for i,r in enumerate(rows) if gold_numeric(r['answer']) is not None]
assert all(good['primary_vector'][i]==good['strict_vector'][i]==1 for i in indices)
assert all(bad['primary_vector'][i]==bad['strict_vector'][i]==0 for i in indices)
for name,r in out['models'].items():
 for t,v in r['sensitivity_vectors'].items():assert v['primary_vector']==r['primary_vector'] and v['strict_vector']==r['strict_vector']
assert not out['parser_comparison_warnings']
write(actual/'audits/minerva_scorer_integration.json',dict(time=time.time(),script_sha256=sha(__file__),scorer_sha256=sha(actual/'scripts/score_minerva_numeric.py'),helper_sha256=sha(actual/'scripts/minerva_numeric.py'),fixture_output_sha256=sha(fixture/'audits/scoring_minerva_numeric_corrected.json'),passed=True,checks='Actualscorerscript:all191numericcorrectaccepted,all191factor2/zero+1wrongrejected;81symbolicoriginal1/strict0preserved;bothsensitivitiesmatchtheseunambiguouscases;0warnings.',scope='Syntheticfixtureonly,noGPU/modelinferenceandnorealfinaloutput.' ))
print('Minerva scorer integration passed; synthetic fixture kept under audits only.')
