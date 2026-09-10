"""Post-outcome local manual scoring sensitivity; never changes frozen score files."""
import ast
import numpy as np
from common import *
pp=S/'audits/ood_manual_sensitivity_plan.json';decl=json.loads(pp.read_text());plan=json.loads((S/'results/confirmation_plan.json').read_text());src=S/'scripts/analyze_confirmation.py'
assert sha(src)==decl['statistics_source_sha256']
ns={'np':np,'seeds':plan['seeds']};nodes=[n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['compare','macro_compare']];assert len(nodes)==2
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
rp=S/'audits/ood_scoring_disagreement_review.json';review=json.loads(rp.read_text());assert review['all_reviewed']
up=S/'audits/minerva_explicit_unit_review.json';units=json.loads(up.read_text());assert units['all_reviewed']
families=dict(plan['families']);families['teacher7_expanded_epoch4']=json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text())['models']
names=['base']+[n for fam in families.values() for n in fam.values()];assert len(set(names))==21
paths=[pp,rp,up,src,S/'results/confirmation_plan.json'];vec={}
num={}
for fn in ['scoring_minerva_numeric_corrected.json','scoring_supplement_minerva_numeric.json']:
 p=S/'audits'/fn;paths.append(p);num.update(json.loads(p.read_text())['models'])
for ds in decl['datasets']:
 scores={}
 for kind in ['confirmation','supplement']:
  p=S/f'audits/scoring_{kind}_{ds}.json';paths.append(p);scores.update(json.loads(p.read_text())['models'])
 vec[ds]={}
 for n in names:
  p=S/'results/evaluation'/ds/n/'math_predictions.jsonl';paths.append(p);pred=read(p);assert sha(p)==scores[n]['predictions_sha256']
  raw=np.array([o['correct'] for o in pred],int);strict=np.array(scores[n]['strict_vector'],int)
  if ds=='ood_minerva':
   assert sha(p)==num[n]['predictions_sha256'];raw=np.array(num[n]['primary_vector'],int);strict=np.array(num[n]['strict_vector'],int)
  vec[ds][n]=(raw,strict)
results={}
for unit_mode in ['disagreement_only','plus_reviewed_percent_answers']:
 results[unit_mode]={}
 for bound in ['effect_lower','effect_upper']:
  corrected={ds:{n:(v[0].copy(),v[1].copy()) for n,v in models.items()} for ds,models in vec.items()}
  for d in review['decisions']:
   value=d['final_answer_correct']
   if value is None:value=(d['model']=='base') if bound=='effect_lower' else (d['model']!='base')
   for v in corrected[d['dataset']][d['model']]:v[d['index']]=int(value)
  if unit_mode=='plus_reviewed_percent_answers':
   for d in units['decisions']:
    if d['final_answer_correct']:
     for o in d['output_bindings']:
      for v in corrected['ood_minerva'][o['model']]:v[d['index']]=1
  for ds,models in corrected.items():
   for n,v in models.items():assert np.array_equal(*v),(ds,n,'unreviewed policy difference')
  out={}
  for family,modelmap in families.items():
   per={};diff=[]
   for ds in decl['datasets']:
    b=corrected[ds]['base'][0];g=np.stack([corrected[ds][modelmap[str(seed)]][0] for seed in plan['seeds']]);d=g-b
    per[ds]=dict(n=len(b),base_correct=int(b.sum()),per_seed_correct=g.sum(1).astype(int).tolist(),vs_base=ns['compare'](d))
    if ds in plan['macro_datasets']:diff.append(d)
   out[family]=dict(datasets=per,ood_macro=ns['macro_compare'](diff))
  results[unit_mode][bound]=out
write(S/'results/ood_manual_scoring_sensitivity.json',dict(time=time.time(),script_sha256=sha(__file__),source_sha256={str(p.relative_to(S)):sha(p) for p in paths},results=results,all_corrected_raw_strict_vectors_equal=True,scope=decl['scope']))
print(json.dumps({u:{b:r[plan['primary_family']]['ood_macro'] for b,r in bs.items()} for u,bs in results.items()},indent=2))
