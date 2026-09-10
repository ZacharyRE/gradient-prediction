"""Apply a predeclared development selection rule, freeze, then run final evidence."""
import subprocess,datetime
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
rp=S/'audits/final_selection_rule.json';rule=json.loads(rp.read_text());rh=sha(rp)
models=['base']+[n for family in rule['families'].values() for n in family.values()];assert len(models)==len(set(models))==16
limit=datetime.datetime(2026,9,10,7,tzinfo=datetime.timezone.utc).timestamp()
def ready():
 p=S/'audits/expanded_seed_replication_dispatch.json'
 return p.exists() and json.loads(p.read_text()).get('all_complete') and (S/'results/matched_generator_control_analysis.json').exists() and all((S/'results/evaluation/dev'/n/'math_summary.json').exists() for n in models)
while time.time()<min(DEADLINE,limit) and not ready():time.sleep(10)
assert ready(),'Final selection readiness cutoff reached'
assert sha(rp)==rh and not (S/'results/confirmation_plan.json').exists()
stat=json.loads((S/'audits/final_statistics_contract.json').read_text());assert stat['passed'] and stat['analysis_script_sha256']==sha(S/'scripts/analyze_confirmation.py')
bridgepath=S/'audits/evaluator_version_bridge.json';bridge=json.loads(bridgepath.read_text());assert bridge['passed']
base=json.loads((S/'results/evaluation/dev/base/math_manifest.json').read_text())
counts={};identities={}
for name in models:
 root=S/'results/evaluation/dev'/name;m=json.loads((root/'math_manifest.json').read_text());r=read(root/'math_predictions.jsonl');summary=json.loads((root/'math_summary.json').read_text())
 assert m['script_sha256'] in bridge['allowed_development_script_sha256']
 for key in ['schema','engine','batch_invariant','max_tokens','temperature','chunk_size','model','data_sha256']:assert m[key]==base[key],(name,key)
 assert len(r)==summary['samples']==500 and sum(bool(x['correct']) for x in r)==summary['correct']
 counts[name]=summary['correct'];identities[name]=dict(predictions_sha256=sha(root/'math_predictions.jsonl'),manifest_sha256=sha(root/'math_manifest.json'))
tag='final_selection_fixed16'
cmd=[sys.executable,str(S/'scripts/score_sensitivity.py'),'--dataset','dev','--tag',tag,'--models',*models]
with (S/'logs/final_selection_strict.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,DEADLINE-time.time()))
sp=S/f'audits/scoring_{tag}.json';strict=json.loads(sp.read_text());assert not strict['parser_comparison_warnings'],'Review parser warnings before applying selection'
strict_counts={}
for name in models:
 m=strict['models'][name];assert m['predictions_sha256']==identities[name]['predictions_sha256'];assert len(m['strict_vector'])==500
 strict_counts[name]=sum(m['strict_vector'])
old=rule['default_primary'];new=rule['alternative_primary'];checks={};snapshots={}
for label,vals in [('raw',counts),('strict',strict_counts)]:
 oldv=[vals[n] for n in rule['families'][old].values()];newv=[vals[n] for n in rule['families'][new].values()]
 checks[label+'_expanded_every_seed_above_baseline']=all(v>vals['base'] for v in newv)
 checks[label+'_expanded_mean_above_old_mean']=sum(newv)>sum(oldv)
 snapshots[label]=dict(base=vals['base'],old_average=oldv,expanded_average=newv,old_epoch4=[vals[n] for n in rule['families']['teacher7_epoch4'].values()])
primary=new if all(checks.values()) else old
selection=dict(primary_family=primary,families=rule['families'],selection_reason='Applied prospectively frozen deterministic development rule; see decision checks. No final benchmark results existed at selection.',control_rationale=rule['controls'],rule_sha256=rh,rule_path=str(rp),decision_checks=checks,development_snapshot=snapshots,development_identities=identities,strict_audit_sha256=sha(sp),statistics_contract_sha256=sha(S/'audits/final_statistics_contract.json'),time=time.time())
selectionpath=S/'audits/final_selection_decision.json';assert not selectionpath.exists();write(selectionpath,selection)
with (S/'logs/final_freeze.log').open('w') as log:subprocess.run([sys.executable,str(S/'scripts/freeze_confirmation.py'),'--selection',str(selectionpath)],stdout=log,stderr=subprocess.STDOUT,check=True)
planpath=S/'results/confirmation_plan.json';ph=sha(planpath);print(json.dumps(dict(event='final_frozen',time=time.time(),primary=primary,checks=checks,scores=snapshots)),flush=True)
processes={};handles=[]
tasks=[('benchmarks','3','final_evaluation_pipeline.py',['--mode','benchmarks']),('native','1','final_evaluation_pipeline.py',['--mode','native']),('collector','','collect_final_evidence.py',[])]
for name,gpu,script,args in tasks:
 handle=(S/f'logs/final_campaign_{name}.log').open('w');handles.append(handle)
 command=[sys.executable,str(S/'scripts'/script),*args]
 process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu));processes[name]=(process,command,gpu)
write(S/'audits/final_campaign_dispatch.json',dict(time=time.time(),plan_sha256=ph,rule_sha256=rh,script_sha256=sha(__file__),children={name:dict(pid=p.pid,command=cmd,gpu=gpu) for name,(p,cmd,gpu) in processes.items()},all_complete=False))
reported={}
while time.time()<DEADLINE:
 for name,(process,command,gpu) in processes.items():
  rc=process.poll()
  if rc is not None and name not in reported:
   reported[name]=rc;print(json.dumps(dict(event='child_finished',name=name,returncode=rc,time=time.time())),flush=True)
 if len(reported)==len(processes):break
 time.sleep(10)
for handle in handles:handle.close()
write(S/'audits/final_campaign_completion.json',dict(time=time.time(),plan_sha256=ph,returncodes=reported,all_complete=len(reported)==3 and all(v==0 for v in reported.values()),scope='Execution/collection completion is not an efficacy claim, parser-warning review, or final report completion.'))
assert len(reported)==3 and all(v==0 for v in reported.values()),reported
