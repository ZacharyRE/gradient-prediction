"""Resume GPU3 evaluation after this study drains, allowing other users' allocations."""
import argparse,signal,subprocess
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')

p=argparse.ArgumentParser();p.add_argument('--owner-pid',type=int,required=True);a=p.parse_args()
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
def proc(pid):
 d=Path('/proc')/str(pid)
 try:
  fields=(d/'stat').read_text().rsplit(')',1)[1].split()
  return dict(uid=d.stat().st_uid,start=fields[19],args=(d/'cmdline').read_bytes().decode(errors='replace').split('\0'),env=(d/'environ').read_bytes().split(b'\0'))
 except (FileNotFoundError,PermissionError,ProcessLookupError):return None
owner=proc(a.owner_pid);assert owner and owner['uid']==os.getuid()
assert any(x.endswith('research/sft_generalization_20260909/scripts/sampling_stability_pipeline.py') for x in owner['args'])
def existing_eval_queue():
 result=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  r=proc(int(d.name))
  if not r or r['uid']!=os.getuid() or b'CUDA_VISIBLE_DEVICES=3' not in r['env']:continue
  if '--queue' in r['args'] and str(S/'scripts/evaluate.py') in r['args']:result.append(int(d.name))
 return result
def own_gpu_processes():
 output=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 result=[]
 for line in output.splitlines():
  if not line.strip().isdigit():continue
  pid=int(line.strip());r=proc(pid)
  if r and r['uid']==os.getuid():result.append(pid)
 return result
def record(status,**extra):
 write(S/'audits/shared_gpu_eval_handoff.json',dict(time=time.time(),status=status,original_owner_pid=a.owner_pid,original_owner_start=owner['start'],gpu=3,script_sha256=sha(__file__),scope='Only the original study scheduler may be signaled, after its child diagnostics return. No other-user process is signaled. Inference code/options unchanged.',**extra))
record('waiting_for_diagnostic_dispatch_completion')
dispatch=S/'audits/sampling_seed_stability_dispatch.json'
while time.time()<DEADLINE and not dispatch.exists():
 if existing_eval_queue():record('original_owner_already_resumed',queue_pids=existing_eval_queue());raise SystemExit(0)
 time.sleep(10)
gpu_guard();dispatch_result=json.loads(dispatch.read_text())
if existing_eval_queue():record('original_owner_already_resumed',queue_pids=existing_eval_queue(),dispatch=dispatch_result);raise SystemExit(0)
live=proc(a.owner_pid);signaled=False
if live and live['uid']==os.getuid() and live['start']==owner['start']:
 os.kill(a.owner_pid,signal.SIGTERM);signaled=True
# Handle the narrow race where the original owner started its evaluator just before SIGTERM.
stable=0
while time.time()<DEADLINE:
 if existing_eval_queue():record('existing_evaluator_retained',queue_pids=existing_eval_queue(),scheduler_signaled=signaled,dispatch=dispatch_result);raise SystemExit(0)
 live=proc(a.owner_pid);owner_alive=bool(live and live['start']==owner['start'])
 owned=own_gpu_processes()
 memory=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
 free,total=[int(x.strip()) for x in memory.strip().split(',')]
 stable=stable+1 if not owner_alive and not owned and free>=.30*total+2048 else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not own_gpu_processes() and not existing_eval_queue()
followup=S/'results/sampling/sample_seed_stability/followup_dispatch.json'
# Resolve the actual sampling-output name from its frozen plan rather than assume it.
resolved=json.loads((S/'results/sampling_seed_stability_resolved.json').read_text())
followup=S/'results/sampling'/resolved['name']/'followup_dispatch.json'
record('resuming_evaluation',scheduler_signaled=signaled,dispatch=dispatch_result,followup_dispatch=json.loads(followup.read_text()) if followup.exists() else None,own_gpu_pids=[],free_memory_mib=free,total_memory_mib=total,consecutive_quiet_checks=stable)
(S/'STOP_EVALUATOR').unlink(missing_ok=True)
command=[sys.executable,str(S/'scripts/evaluate.py'),'--queue','--manifest',str(S/'results/protocol_restart_after_sampling.json')]
with (S/'logs/evaluation_after_shared_handoff.log').open('w') as log:
 result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
record('evaluator_exited',returncode=result.returncode,scheduler_signaled=signaled)

raise SystemExit(result.returncode)
