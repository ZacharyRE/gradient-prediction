"""Separate post-outcome correction of the agreed-but-wrong baseline4060."""
import ast
import numpy as np
from common import *
files=['audits/math4060_operator_gold_followup_plan.json','audits/math4060_operator_gold_review.json','audits/math_scoring_disagreement_review_decisions.json','results/math_manual_disagreement_sensitivity.json','results/confirmation_plan.json','scripts/analyze_confirmation.py','audits/math_operator_semantics_revision.json']
plan=json.loads((S/files[4]).read_text()); prior=json.loads((S/files[3]).read_text())
for p,d in prior['source_sha256'].items(): assert sha(S/p)==d,p
review=json.loads((S/files[1]).read_text()); decisions=json.loads((S/files[2]).read_text())
assert review['all_reviewed'] and review['n']==21 and decisions['all_reviewed']
assert review['plan_sha256']==sha(S/files[0])
assert review['semantic_revision_sha256']==sha(S/files[-1])==decisions['semantic_revision_sha256']
assert sum(r['mathematical_final_answer_correct'] is None for r in review['records'])==7
source=S/files[5]; assert sha(source)==plan['analysis_script_sha256']
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
names=['base']+[plan['families'][plan['primary_family']][str(i)] for i in plan['seeds']]
raw={n:np.array([r['correct'] for r in read(S/'results/evaluation/math_reused5000'/n/'math_predictions.jsonl')],int) for n in names}
result={}
for interpretation in ['answer_markup','literal_divisor_operator']:
 result[interpretation]={}
 for bound in ['conservative_lower','conservative_upper']:
  v={n:x.copy() for n,x in raw.items()}
  for d in decisions['decisions']:
   n=d['model']; value=d['mathematical_final_answer_correct']
   if value is None:value=(n=='base') if bound=='conservative_lower' else (n!='base')
   v[n][d['index']]=int(value)
  for n in names:
   row=next(r for r in review['records'] if r['model']==n)
   assert row['predictions_sha256']==sha(S/'results/evaluation/math_reused5000'/n/'math_predictions.jsonl')
   v[n][4060]=int(row[interpretation+'_interpretation_correct'])
  for part,k in [('full5000',np.arange(5000)),('remaining3500',np.array(plan['validation_excluded_sensitivity']['keep_indices']))]:
   b=v['base'][k];g=np.stack([v[n][k] for n in names[1:]])
   result[interpretation].setdefault(part,{})[bound]={'n':len(k),'base_correct':int(b.sum()),'per_seed_correct':g.sum(1).astype(int).tolist(),'vs_base':ns['compare'](g-b)}
out={'time':time.time(),'script_sha256':sha(__file__),'source_sha256':{p:sha(S/p) for p in files},'results_by_interpretation':result,'scope':'Additional post-outcome partial manual sensitivity after explicit semantic revision. Use each of two globallyconsistent boxed conventions for all21cases includingbaseline, while independently bounding the4nonoperator incomplete/contradictorycases. Literalboxed12 is divisorcount12=6 and correctasexpression despiteflawedprocess; answer-markupboxed12 isnumeric12andwrong. Other14outputswrongundereither. Recompute pairedbootstrap from frozencompare; otheragreedjudgments notcertified andoriginalcriteriaunchanged.'}
write(S/'results/math4060_followup_sensitivity.json',out);print(json.dumps(result,indent=2))
