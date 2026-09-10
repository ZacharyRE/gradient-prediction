"""Explicit, non-signaling ownership handoff after final model selection is frozen."""
import argparse, subprocess
from common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['benchmarks','native'],required=True);a=p.parse_args()
 os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
 gpu_guard();gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu==('3' if a.mode=='benchmarks' else '1')
 planpath=S/'results/confirmation_plan.json';plan=json.loads(planpath.read_text());planhash=sha(planpath)
 assert sha(S/'scripts/evaluate.py')==plan['evaluation_protocol']['script_sha256']
 assert sha(S/'results/confirmation_manifest.json')==plan['evaluation_manifest_sha256']
 assert sha(S/'scripts/final_evaluation_pipeline.py')==plan['final_pipeline_script_sha256']
 for d in plan['datasets']:assert sha(S/f'data/{d}.jsonl')==plan['data_sha256'][d]
 sentinel=S/('STOP_EVALUATOR' if a.mode=='benchmarks' else 'STOP_TRAINER')
 sentinel.touch()
 def live_old_workers():
  names={'evaluate.py'} if a.mode=='benchmarks' else {'train.py','train_anchored.py','train_queue.py'}
  result=[]
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    if proc.stat().st_uid!=os.getuid():continue
    if f'CUDA_VISIBLE_DEVICES={gpu}'.encode() not in (proc/'environ').read_bytes().split(b'\0'):continue
    args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    if any(Path(x).name in names and Path(x).resolve().parent==S/'scripts' for x in args if x.endswith('.py')):result.append(int(proc.name))
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  return result
 def own_gpu_pids():
  result=[]
  for line in subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
   if not line.strip().isdigit():continue
   try:
    if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():result.append(int(line))
   except FileNotFoundError:pass
  return result
 def drain(initial=False):
  stable=0;checks=[]
  while time.time()<DEADLINE:
   workers=live_old_workers() if initial else [];owned=own_gpu_pids()
   free,total=map(int,subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
   required=.30*total+2048 if a.mode=='benchmarks' else 45*1024
   good=not workers and not owned and free>=required
   stable=stable+1 if good else 0
   checks.append(dict(time=time.time(),workers=workers,owned_gpu_pids=owned,free_mib=free,required_mib=required,quiet=good))
   if stable>=2:return checks
   time.sleep(5)
  raise RuntimeError('Deadline reached while waiting for natural GPU ownership drain')
 checks=drain(initial=True)
 write(S/f'audits/final_{a.mode}_handoff.json',dict(time=time.time(),gpu=gpu,checks=checks,plan_sha256=planhash,script_sha256=sha(__file__),scope='Current worker finished naturally through STOP sentinel; no process was signaled. Foreign allocations untouched. Sentinel remains set throughout final work.'))
 if a.mode=='benchmarks':
  # Fill any missing development replicas only after recipe selection is immutable.
  tasks=[(d,[sys.executable,str(S/'scripts/evaluate.py'),'--dataset',d,'--manifest',str(S/'results/confirmation_manifest.json')]) for d in ['dev']+plan['datasets']]
 else:
  nativeplan=S/'audits/final_native_protocol_plan.json'
  assert sha(nativeplan)==plan['native_protocol_plan_sha256']
  np=json.loads(nativeplan.read_text());assert sha(S/'scripts/native_final_recheck.py')==np['native_script_sha256']
  allmodels=json.loads((S/'results/confirmation_manifest.json').read_text())
  models={'base':None,**{name:allmodels[name] for name in plan['families'][plan['primary_family']].values()}}
  path=S/'results/final_native_manifest.json'
  if path.exists():assert json.loads(path.read_text())==models
  else:write(path,models)
  tasks=[('dev500',[sys.executable,str(S/'scripts/native_final_recheck.py'),'--manifest',str(path),'--limit','500','--batch','8'])]
 dispatch=[]
 for label,command in tasks:
  gpu_guard();assert sha(planpath)==planhash
  logpath=S/f'logs/final_{a.mode}_{label}.log'
  with logpath.open('a') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  dispatch.append(dict(time=time.time(),label=label,command=command,returncode=result.returncode,log=str(logpath)))
  write(S/f'audits/final_{a.mode}_dispatch.json',dict(plan_sha256=planhash,tasks=dispatch,all_complete=False))
  if result.returncode:raise SystemExit(result.returncode)
  drain()
 write(S/f'audits/final_{a.mode}_dispatch.json',dict(plan_sha256=planhash,tasks=dispatch,all_complete=True,time=time.time()))

if __name__=='__main__':main()
