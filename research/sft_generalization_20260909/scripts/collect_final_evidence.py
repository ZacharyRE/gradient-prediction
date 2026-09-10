"""CPU-only collection after immutable final selection; never launches GPU work."""
import subprocess
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Collector is CPU only'
planpath=S/'results/confirmation_plan.json';plan=json.loads(planpath.read_text());planhash=sha(planpath)
assert sha(__file__)==plan['final_collector_script_sha256']
models=list(json.loads((S/'results/confirmation_manifest.json').read_text()));primary=['base']+list(plan['families'][plan['primary_family']].values())
calls=[];scored=set();done=set();numeric_done=False
def run(script,args,label):
 gpu_disabled=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 command=[sys.executable,str(S/'scripts'/script),*args]
 with (S/f'logs/collector_final_{label}.log').open('a') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=gpu_disabled,timeout=max(1,DEADLINE-time.time()))
 calls.append(dict(time=time.time(),label=label,command=command,returncode=result.returncode))
 write(S/'audits/final_collection_calls.json',dict(plan_sha256=planhash,calls=calls))
 assert result.returncode==0,(label,result.returncode)
def eval_complete(dataset,names):return all((S/'results/evaluation'/dataset/n/'math_summary.json').exists() for n in names)
def checked_analysis(path):
 if not path.exists():return False
 r=json.loads(path.read_text());assert r['plan_sha256']==planhash;return True
while time.time()<DEADLINE and not (S/'STOP_FINAL_COLLECTOR').exists():
 assert sha(planpath)==planhash
 if eval_complete('math_reused5000',models) and 'replay' not in done:
  p=S/'audits/final_math_overlap_replay.json'
  if not checked_analysis(p):run('audit_final_math_replay.py',[],'math_replay')
  assert json.loads(p.read_text())['passed'];done.add('replay')
 for dataset in plan['datasets']:
  if dataset in scored or not eval_complete(dataset,models):continue
  p=S/f'audits/scoring_confirmation_{dataset}.json'
  if not p.exists():run('score_sensitivity.py',['--dataset',dataset,'--tag','confirmation_'+dataset,'--models',*models],'strict_'+dataset)
  r=json.loads(p.read_text());assert r['script_sha256']==plan['strict_scoring_script_sha256'] and r['dataset_sha256']==plan['data_sha256'][dataset]
  assert set(r['models'])==set(models)
  for name in models:assert r['models'][name]['predictions_sha256']==sha(S/'results/evaluation'/dataset/name/'math_predictions.jsonl')
  scored.add(dataset)
 if 'ood_minerva' in scored and not numeric_done:
  p=S/'audits/scoring_minerva_numeric_corrected.json'
  if not p.exists():run('score_minerva_numeric.py',[],'minerva_numeric_corrected')
  r=json.loads(p.read_text());assert r['plan_sha256']==planhash and r['script_sha256']==plan['minerva_numeric_scoring_script_sha256']
  assert r['helper_sha256']==plan['minerva_numeric_helper_sha256'] and r['contract_sha256']==plan['minerva_numeric_contract_sha256']
  assert set(r['models'])==set(models)
  numeric_done=True
 if all(eval_complete(d,models) for d in plan['datasets']) and numeric_done:
  for label,args,path in [('main',[],S/'results/confirmation_analysis.json'),('main_strict',['--strict'],S/'results/confirmation_analysis_strict.json')]:
   if label in done:continue
   if label=='main_strict' and len(scored)!=len(plan['datasets']):continue
   if not checked_analysis(path):run('analyze_confirmation.py',args,label)
   assert checked_analysis(path);done.add(label)
 native_ready=all((S/'results/native/final_dev500'/n/'summary.json').exists() for n in primary)
 if native_ready and eval_complete('dev',primary):
  p=S/'audits/final_native_comparison.json'
  if 'native' not in done:
   if not checked_analysis(p):run('analyze_final_native.py',[],'native')
   assert checked_analysis(p);done.add('native')
  if 'native_strict' not in done:
   run('prepare_final_native_strict.py',[],'prepare_native_strict')
   p=S/'audits/final_native_comparison_strict.json'
   if not checked_analysis(p):run('analyze_final_native.py',['--strict'],'native_strict')
   assert checked_analysis(p);done.add('native_strict')
 complete=done=={'replay','main','main_strict','native','native_strict'} and len(scored)==len(plan['datasets']) and numeric_done
 write(S/'audits/final_collection_status.json',dict(time=time.time(),plan_sha256=planhash,scored_datasets=sorted(scored),analyses_complete=sorted(done),minerva_numeric_corrected=numeric_done,all_complete=complete,scope='Collection complete does not mean efficacy criteria passed or final report/case-warning review completed.'))
 if complete:break
 time.sleep(10)
