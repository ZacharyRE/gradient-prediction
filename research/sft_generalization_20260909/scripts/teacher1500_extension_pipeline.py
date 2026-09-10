"""Bounded four-model validation detour, with explicit GPU3 ownership drain."""
import subprocess
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
plan=json.loads((S/'audits/teacher_five_seed_math1500_extension_plan.json').read_text())
def active_queue():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if '--queue' not in args or b'CUDA_VISIBLE_DEVICES=3' not in (p/'environ').read_bytes().split(b'\0'):continue
   if any(Path(a).resolve()==(S/'scripts/evaluate.py').resolve() for a in args if a.endswith('.py')):result.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return result
def own_gpu_pids():
 result=[]
 for line in subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
  if not line.strip().isdigit():continue
  try:
   if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():result.append(int(line))
  except FileNotFoundError:pass
 return result
while time.time()<DEADLINE:
 if all((S/'results/evaluation/dev'/name/'math_summary.json').exists() for name in plan['wait_for_dev_primary']):break
 time.sleep(10)
gpu_guard();(S/'STOP_EVALUATOR').touch();stable=0
while time.time()<DEADLINE:
 free,total=map(int,subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
 stable=stable+1 if not active_queue() and not own_gpu_pids() and free>=.30*total+2048 else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not active_queue() and not own_gpu_pids()
write(S/'audits/teacher1500_extension_handoff.json',dict(time=time.time(),quiet_checks=stable,own_gpu_pids=[],free_memory_mib=free,script_sha256=sha(__file__),scope='No process signaled; existing dev queue finished its current model and exited. Foreign allocations untouched.'))
command=[sys.executable,str(S/'scripts/evaluate.py'),'--dataset','math_reused1500','--manifest',str(S/'results/teacher_five_seed_math1500_extension_manifest.json')]
with (S/'logs/evaluation_teacher1500_extension.log').open('w') as log:
 r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/teacher1500_extension_dispatch.json',dict(time=time.time(),returncode=r.returncode,command=command))
stable=0
while time.time()<DEADLINE:
 stable=stable+1 if not own_gpu_pids() else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not own_gpu_pids() and not active_queue();(S/'STOP_EVALUATOR').unlink()
command=[sys.executable,str(S/'scripts/evaluate.py'),'--queue','--manifest',str(S/'results/protocol_restart_after_teacher1500_extension.json')]
with (S/'logs/evaluation_after_teacher1500_extension.log').open('w') as log:
 result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/teacher1500_extension_queue_exit.json',dict(time=time.time(),returncode=result.returncode,extension_returncode=r.returncode))
raise SystemExit(result.returncode or r.returncode)
