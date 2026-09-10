import argparse
import numpy as np
from scipy.stats import binomtest
from common import *
from gradient_geometry.sft_protocol import sample_identity
p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');a=p.parse_args()
plan_path=S/'results/confirmation_plan.json';plan=json.loads(plan_path.read_text());seeds=plan['seeds'];families=plan['families'];datasets=plan['datasets'];results={};vectors={}
assert seeds==[43,44,45,46,47]
assert sha(S/'results/confirmation_manifest.json')==plan['evaluation_manifest_sha256']
assert sha(__file__)==plan['analysis_script_sha256']
strict={}
numeric_path=S/'audits/scoring_minerva_numeric_corrected.json';numeric=json.loads(numeric_path.read_text())
assert numeric['plan_sha256']==sha(plan_path) and numeric['script_sha256']==plan['minerva_numeric_scoring_script_sha256']
assert numeric['helper_sha256']==plan['minerva_numeric_helper_sha256'] and numeric['contract_sha256']==plan['minerva_numeric_contract_sha256']
assert numeric['dataset_sha256']==plan['data_sha256']['ood_minerva']
if a.strict:
 for dataset in datasets:
  audit_path=S/f'audits/scoring_confirmation_{dataset}.json';audit=json.loads(audit_path.read_text())
  assert audit['dataset_sha256']==plan['data_sha256'][dataset]
  assert audit['script_sha256']==plan['strict_scoring_script_sha256']
  strict[dataset]=audit
def scores(dataset,name,rows):
 if dataset=='ood_minerva':
  record=numeric['models'][name];assert record['predictions_sha256']==sha(S/'results/evaluation'/dataset/name/'math_predictions.jsonl')
  v=record['strict_vector' if a.strict else 'primary_vector'];assert len(v)==len(rows)
  return np.array(v,int)
 if not a.strict:return np.array([x['correct'] for x in rows],int)
 record=strict[dataset]['models'][name]
 assert record['predictions_sha256']==sha(S/'results/evaluation'/dataset/name/'math_predictions.jsonl')
 assert len(record['strict_vector'])==len(rows)
 return np.array(record['strict_vector'],int)
def compare(d):
 rng=np.random.default_rng(20260911);ns,n=d.shape;mean=d.mean(0)
 boots=np.concatenate([mean[rng.integers(n,size=(200,n))].mean(1) for _ in range(50)])*100
 cross=[]
 for _ in range(10000):cross.append(d[rng.integers(ns,size=ns)][:,rng.integers(n,size=n)].mean()*100)
 return dict(delta_pp=float(d.mean()*100),question_ci95=np.quantile(boots,[.025,.975]).tolist(),seed_question_ci95=np.quantile(cross,[.025,.975]).tolist(),per_seed_delta_pp=(d.mean(1)*100).tolist(),every_seed_positive=bool(np.all(d.mean(1)>0)))
for dataset in datasets:
 src=S/f'data/{dataset}.jsonl';assert sha(src)==plan['data_sha256'][dataset]
 root=S/'results/evaluation'/dataset;base=read(root/'base/math_predictions.jsonl');assert (root/'base/math_summary.json').exists();b=scores(dataset,'base',base);hashes=[r['sample_hash'] for r in base];res=dict(n=len(base),base_correct=int(b.sum()),base_accuracy=float(b.mean()*100),families={},comparisons={});vectors[dataset]={}
 baseline_manifest=json.loads((root/'base/math_manifest.json').read_text())
 assert baseline_manifest['data_sha256']==sha(src) and len(base)==len(read(src))
 assert hashes==[sample_identity(r) for r in read(src)]
 if 'base_model_files_sha256' in plan:assert baseline_manifest['model']['files']==plan['base_model_files_sha256']
 for key,value in plan['evaluation_protocol'].items():assert baseline_manifest[key]==value,(dataset,key)
 for family,mapping in families.items():
  group=[];per={}
  for seed in seeds:
   name=mapping[str(seed)];assert (root/name/'math_summary.json').exists();rr=read(root/name/'math_predictions.jsonl');assert len(rr)==len(base) and [r['sample_hash'] for r in rr]==hashes
   manifest=json.loads((root/name/'math_manifest.json').read_text());assert manifest['data_sha256']==sha(src);assert manifest['adapter']['files']['adapter_model.safetensors']==plan['adapter_sha256'][name]
   assert manifest['model']==baseline_manifest['model']
   assert manifest['adapter']['files']==plan['adapter_files_sha256'][name]
   for key,value in plan['evaluation_protocol'].items():assert manifest[key]==value,(dataset,name,key)
   v=scores(dataset,name,rr);group.append(v);w=int(((v==1)&(b==0)).sum());l=int(((v==0)&(b==1)).sum());per[str(seed)]=dict(correct=int(v.sum()),accuracy=float(v.mean()*100),wrong_to_right=w,right_to_wrong=l,mcnemar_p=binomtest(w,w+l).pvalue if w+l else 1.,vs_base=compare((v-b)[None,:]))
  group=np.stack(group);vectors[dataset][family]=group-b;res['families'][family]=dict(mean_accuracy=float(group.mean()*100),per_seed=per,vs_base=compare(group-b))
 for control in families:
  if control!=plan['primary_family']:res['comparisons'][plan['primary_family']+'_vs_'+control]=compare(vectors[dataset][plan['primary_family']]-vectors[dataset][control])
 results[dataset]=res
macro={};ood=plan['macro_datasets']
def macro_compare(ds):
 rng=np.random.default_rng(20260912);point=np.stack([d.mean(1) for d in ds]).mean(0)*100;boots=[];cross=[]
 for _ in range(10000):
  si=rng.integers(len(seeds),size=len(seeds));sampled=[d[:,rng.integers(d.shape[1],size=d.shape[1])] for d in ds]
  boots.append(np.mean([d.mean() for d in sampled])*100);cross.append(np.mean([d[si].mean() for d in sampled])*100)
 return dict(delta_pp=float(point.mean()),per_seed_delta_pp=point.tolist(),every_seed_positive=bool(np.all(point>0)),question_ci95=np.quantile(boots,[.025,.975]).tolist(),seed_question_ci95=np.quantile(cross,[.025,.975]).tolist())
for family in families:macro[family]=macro_compare([vectors[name][family] for name in ood])
numeric_sensitivity={}
for tolerance in ['legacy_saved_scoring',*numeric['models']['base']['sensitivity_vectors']]:
 alternate={}
 for family,mapping in families.items():
  if tolerance=='legacy_saved_scoring':
   if a.strict:
    old=json.loads((S/'audits/scoring_confirmation_ood_minerva.json').read_text())['models'];bv=np.array(old['base']['strict_vector'],int)
    group=np.stack([old[mapping[str(seed)]]['strict_vector'] for seed in seeds])
   else:
    folder=S/'results/evaluation/ood_minerva';bv=np.array([r['correct'] for r in read(folder/'base/math_predictions.jsonl')],int)
    group=np.stack([[r['correct'] for r in read(folder/mapping[str(seed)]/'math_predictions.jsonl')] for seed in seeds])
  else:
   field='strict_vector' if a.strict else 'primary_vector';bv=np.array(numeric['models']['base']['sensitivity_vectors'][tolerance][field],int)
   group=np.stack([numeric['models'][mapping[str(seed)]]['sensitivity_vectors'][tolerance][field] for seed in seeds])
  d=group-bv
  alternate[family]=dict(minerva_base_correct=int(bv.sum()),minerva_per_seed_correct=group.sum(1).astype(int).tolist(),minerva_vs_base=compare(d),ood_macro=macro_compare([d if dataset=='ood_minerva' else vectors[dataset][family] for dataset in ood]))
 numeric_sensitivity[tolerance]=alternate
macro_controls={plan['primary_family']+'_vs_'+control:macro_compare([vectors[name][plan['primary_family']]-vectors[name][control] for name in ood]) for control in families if control!=plan['primary_family']}
primary=plan['primary_family'];idres=results[plan['primary_dataset']]['families'][primary]['vs_base'];oodres=macro[primary]
sensitivity={}
for dataset,excluded in plan.get('sensitivity_exclusions',{}).items():
 assert dataset in vectors
 br=read(S/'results/evaluation'/dataset/'base/math_predictions.jsonl');keep=np.array([i not in set(excluded) for i in range(len(br))]);assert int(keep.sum())==len(br)-len(set(excluded))
 sensitivity[dataset]=dict(n=int(keep.sum()),excluded_indices=excluded,base_correct=int(scores(dataset,'base',br)[keep].sum()),families={family:compare(d[:,keep]) for family,d in vectors[dataset].items()},scope='Prospectively declared conservative candidate-overlap sensitivity; excluded rows are not automatically confirmed leakage.')
holdout=plan['validation_excluded_sensitivity'];dataset=holdout['dataset'];keep=np.array(holdout['keep_indices'],int)
numeric_math={}
numeric_math_plan=plan['math_numeric_grader_sensitivity'];ndataset=numeric_math_plan['dataset']
nbrows=read(S/'results/evaluation'/ndataset/'base/math_predictions.jsonl');nb=scores(ndataset,'base',nbrows)
for label,excluded in numeric_math_plan['sets'].items():
 nk=np.array([i for i in range(len(nbrows)) if i not in set(excluded)],int)
 numeric_math[label]=dict(n=len(nk),excluded_indices=excluded,base_correct=int(nb[nk].sum()),families={family:compare(d[:,nk]) for family,d in vectors[ndataset].items()},scope=numeric_math_plan['scope'])
assert len(keep)==len(set(keep.tolist()))==3500 and holdout['data_sha256']==plan['data_sha256'][dataset]
br=read(S/'results/evaluation'/dataset/'base/math_predictions.jsonl');b=scores(dataset,'base',br)
selection_holdout=dict(n=len(keep),base_correct=int(b[keep].sum()),families={family:dict(vs_base=compare(d[:,keep]),per_seed_correct={str(seed):int((d[j,keep]+b[keep]).sum()) for j,seed in enumerate(seeds)}) for family,d in vectors[dataset].items()},controls={primary+'_vs_'+control:compare((vectors[dataset][primary]-vectors[dataset][control])[:,keep]) for control in families if control!=primary},scope=holdout['scope'])
destination=S/('results/confirmation_analysis_strict.json' if a.strict else 'results/confirmation_analysis.json')
write(destination,dict(plan_sha256=sha(plan_path),script_sha256=sha(__file__),scoring='strict_last_box_sensitivity' if a.strict else 'original_prespecified',datasets=results,ood_macro=macro,ood_macro_controls=macro_controls,minerva_numeric_sensitivity=numeric_sensitivity,math_numeric_grader_sensitivity=numeric_math,minerva_numeric_audit_sha256=sha(numeric_path),sensitivity=sensitivity,validation_excluded_sensitivity=selection_holdout,criteria=dict(primary_every_seed_above_base=idres['every_seed_positive'],primary_mean_ci_above_zero=idres['question_ci95'][0]>0,primary_seed_question_ci_above_zero=idres['seed_question_ci95'][0]>0,ood_every_seed_above_base=oodres['every_seed_positive'],ood_mean_ci_above_zero=oodres['question_ci95'][0]>0,ood_seed_question_ci_above_zero=oodres['seed_question_ci95'][0]>0),notes=['Minerva191numeric answers use prospectively repaired scientifically parsed relative1e-4/atol0 lastboxed scoring; savedlegacy plus1%/5% tolerance analyses are sensitivity only.','All old MATH confirmation data are reused validation, not untouched new tests.','OOD macro uses prespecified equally weighted mathematical datasets, not general nonmathematical capability.','Five fixed seeds provide finite-run evidence, not a guarantee for every possible seed.','Control contrasts are secondary and their unadjusted intervals do not account for all exploratory selection.']))
print(json.dumps(dict(primary=idres,ood_macro=oodres),indent=2))
