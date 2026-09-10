"""Post-outcome reference-quality sensitivity, never a replacement efficacy rule."""
import ast
import numpy as np
from common import *
pp=S/'audits/minerva_pendulum_reference_sensitivity_plan.json'; declaration=json.loads(pp.read_text())
fp=S/'results/confirmation_plan.json';plan=json.loads(fp.read_text());src=S/'scripts/analyze_confirmation.py'
assert sha(fp)==declaration['primary_plan_sha256'] and sha(src)==declaration['statistics_source_sha256']
assert sha(S/'audits/minerva_main_warning_preview_review.json')==declaration['reference_review_sha256']
nodes=[n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['compare','macro_compare']]
assert len(nodes)==2
ns={'np':np,'seeds':plan['seeds']};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
names=['base']+[plan['families'][plan['primary_family']][str(i)] for i in plan['seeds']]
paths=[pp,fp,src,S/'audits/minerva_main_warning_preview_review.json',S/'audits/scoring_minerva_numeric_corrected.json']
numeric=json.loads(paths[-1].read_text());results={}
for mode in ['raw','strict']:
 ap=S/('results/confirmation_analysis.json' if mode=='raw' else 'results/confirmation_analysis_strict.json')
 original=json.loads(ap.read_text());assert original['plan_sha256']==sha(fp);paths.append(ap)
 ds=[]
 for dataset in plan['macro_datasets']:
  vectors=[]
  sp=S/f'audits/scoring_confirmation_{dataset}.json';scoring=json.loads(sp.read_text());paths.append(sp)
  for n in names:
   p=S/'results/evaluation'/dataset/n/'math_predictions.jsonl';paths.append(p)
   rows=read(p);assert scoring['models'][n]['predictions_sha256']==sha(p)
   if dataset=='ood_minerva':
    assert numeric['models'][n]['predictions_sha256']==sha(p)
    v=np.array(numeric['models'][n]['primary_vector' if mode=='raw' else 'strict_vector'],int)
   elif mode=='strict':v=np.array(scoring['models'][n]['strict_vector'],int)
   else:v=np.array([r['correct'] for r in rows],int)
   vectors.append(v)
  b=vectors[0];g=np.stack(vectors[1:]);d=g-b
  if dataset=='ood_minerva':
   keep=np.array([i for i in range(len(b)) if i not in declaration['excluded_indices']]);assert len(keep)==271
   result={'n':271,'base_correct':int(b[keep].sum()),'per_seed_correct':g[:,keep].sum(1).astype(int).tolist(),'vs_base':ns['compare'](d[:,keep])}
   d=d[:,keep]
  ds.append(d)
 results[mode]={'original_minerva':original['datasets']['ood_minerva']['families'][plan['primary_family']]['vs_base'],'original_ood_macro':original['ood_macro'][plan['primary_family']],'minerva_excluding132':result,'ood_macro_excluding132':ns['macro_compare'](ds)}
out={'time':time.time(),'script_sha256':sha(__file__),'plan_sha256':sha(pp),'source_sha256':{str(p.relative_to(S)):sha(p) for p in paths},'results':results,'scope':declaration['scope']}
write(S/'results/minerva_pendulum_reference_sensitivity.json',out);print(json.dumps(results,indent=2))
