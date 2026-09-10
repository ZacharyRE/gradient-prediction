"""Run the actual final analyzer on synthetic fixtures to verify repaired metric routing."""
import runpy,shutil
import common
from common import *
from gradient_geometry.sft_protocol import sample_identity
actual=S;fixture=S/'audits/final_analysis_synthetic_fixture';assert not fixture.exists();fixture.mkdir()
for d in ['scripts','data','audits','results']:(fixture/d).mkdir(exist_ok=True)
datasets=['math_reused5000','ood_minerva','ood_olympiad','ood_svamp','ood_amc23'];seeds=[43,44,45,46,47]
families={f:{str(seed):f'synthetic_{f}_{seed}' for seed in seeds} for f in ['numeric_primary','zero_control','same_numeric_control']}
names=['base']+[n for mapping in families.values() for n in mapping.values()]
data_sha={}
for d in datasets:
 shutil.copy2(actual/f'data/{d}.jsonl',fixture/f'data/{d}.jsonl');data_sha[d]=sha(fixture/f'data/{d}.jsonl')
write(fixture/'results/confirmation_manifest.json',{n:None if n=='base' else 'SYNTHETIC_NO_MODEL' for n in names})
protocol=dict(schema=1,script_sha256='SYNTHETIC_EVALUATOR',engine={'synthetic':True},batch_invariant=True,max_tokens=2048,temperature=0,chunk_size=500)
files={'adapter_model.safetensors':'SYNTHETIC_NO_WEIGHTS'};model={'files':{'synthetic':'NO_MODEL'}}
holdout=json.loads((actual/'audits/math_without_current_validation_plan.json').read_text())
np=json.loads((actual/'audits/math_numeric_grader_sensitivity_plan.json').read_text())
plan=dict(seeds=seeds,families=families,datasets=datasets,macro_datasets=datasets[1:4],primary_family='numeric_primary',primary_dataset='math_reused5000',evaluation_manifest_sha256=sha(fixture/'results/confirmation_manifest.json'),analysis_script_sha256=sha(actual/'scripts/analyze_confirmation.py'),strict_scoring_script_sha256=sha(actual/'scripts/score_sensitivity.py'),data_sha256=data_sha,evaluation_protocol=protocol,base_model_files_sha256=model['files'],adapter_sha256={n:files['adapter_model.safetensors'] for n in names[1:]},adapter_files_sha256={n:files for n in names[1:]},minerva_numeric_scoring_script_sha256=sha(actual/'scripts/score_minerva_numeric.py'),minerva_numeric_helper_sha256=sha(actual/'scripts/minerva_numeric.py'),minerva_numeric_contract_sha256=sha(actual/'audits/minerva_numeric_contract.json'),validation_excluded_sensitivity=holdout,math_numeric_grader_sensitivity=np,sensitivity_exclusions={'math_reused5000':[2121,2432,4678]},scope='SYNTHETIC_STATISTICS_TEST_NOT_MODEL_PERFORMANCE')
write(fixture/'results/confirmation_plan.json',plan);ph=sha(fixture/'results/confirmation_plan.json');numeric_models={}
numeric_indices={r['index'] for r in json.loads((actual/'audits/minerva_numeric_contract.json').read_text())['numeric_references']}
for dataset in datasets:
 rows=read(fixture/f'data/{dataset}.jsonl');strict_models={}
 for name in names:
  out=fixture/'results/evaluation'/dataset/name;out.mkdir(parents=True)
  predictions=[dict(index=i,sample_hash=sample_identity(r),correct=False,prediction='SYNTHETIC_NOT_MODEL_OUTPUT',finish_reason='stop',generated_tokens=1) for i,r in enumerate(rows)]
  jsonl(out/'math_predictions.jsonl',predictions);write(out/'math_summary.json',dict(samples=len(rows),correct=0,synthetic=True))
  write(out/'math_manifest.json',dict(**protocol,data_sha256=data_sha[dataset],model=model,adapter=None if name=='base' else {'files':files}))
  strict_models[name]=dict(strict_vector=[0]*len(rows),predictions_sha256=sha(out/'math_predictions.jsonl'))
  if dataset=='ood_minerva':
   positive=name.startswith('synthetic_numeric_primary') or name.startswith('synthetic_same_numeric_control')
   v=[int(positive and i in numeric_indices) for i in range(len(rows))]
   numeric_models[name]=dict(n=len(rows),predictions_sha256=sha(out/'math_predictions.jsonl'),primary_vector=v,strict_vector=v,sensitivity_vectors={str(t):dict(primary_vector=v,strict_vector=v) for t in [.01,.05]})
 write(fixture/f'audits/scoring_confirmation_{dataset}.json',dict(script_sha256=plan['strict_scoring_script_sha256'],dataset_sha256=data_sha[dataset],models=strict_models))
write(fixture/'audits/scoring_minerva_numeric_corrected.json',dict(plan_sha256=ph,script_sha256=plan['minerva_numeric_scoring_script_sha256'],helper_sha256=plan['minerva_numeric_helper_sha256'],contract_sha256=plan['minerva_numeric_contract_sha256'],dataset_sha256=data_sha['ood_minerva'],models=numeric_models))
common.S=fixture;old_argv=list(sys.argv);outputs=[]
try:
 for flag,suffix in [([],''),(['--strict'],'_strict')]:
  sys.argv=[str(actual/'scripts/analyze_confirmation.py'),*flag]
  runpy.run_path(str(actual/'scripts/analyze_confirmation.py'),run_name='__main__')
  out=fixture/f'results/confirmation_analysis{suffix}.json';r=json.loads(out.read_text())
  expected=191/272*100/3
  assert abs(r['ood_macro']['numeric_primary']['delta_pp']-expected)<1e-10
  assert r['ood_macro']['zero_control']['delta_pp']==0
  assert r['ood_macro_controls']['numeric_primary_vs_same_numeric_control']['delta_pp']==0
  assert r['minerva_numeric_sensitivity']['legacy_saved_scoring']['numeric_primary']['ood_macro']['delta_pp']==0
  assert r['datasets']['ood_minerva']['families']['numeric_primary']['per_seed']['43']['correct']==191
  assert r['validation_excluded_sensitivity']['n']==3500
  assert r['math_numeric_grader_sensitivity']['numeric_false_accept_candidate']['n']==4999
  assert r['math_numeric_grader_sensitivity']['numeric_plus_conservative_overlap']['n']==4996
  assert not r['criteria']['primary_every_seed_above_base'] and r['criteria']['ood_every_seed_above_base']
  outputs.append(dict(scoring=suffix or 'primary',output_sha256=sha(out),expected_ood_macro_pp=expected,actual_ood_macro_pp=r['ood_macro']['numeric_primary']['delta_pp']))
finally:common.S=actual;sys.argv=old_argv
write(actual/'audits/final_analysis_integration.json',dict(time=time.time(),script_sha256=sha(__file__),analysis_script_sha256=sha(actual/'scripts/analyze_confirmation.py'),passed=True,outputs=outputs,scope='Actualraw/strictanalyzers onsyntheticfixtures only. Known191numericrescuesmustpropagateasone-thirdtaskmacro;legacy0mustremain0;unchangedMATHnullmustfailprimarycriteria. Notrealmodelperformance. Fixturesunder audits only.'))
print('Final analysis metric-routing integration passed.',flush=True)
