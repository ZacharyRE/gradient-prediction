"""Drain GPU1 training, generate matched7B expansion, then resume unchanged queue."""
import subprocess
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1' and (S/'STOP_TRAINER').exists()
def owned_training():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if b'CUDA_VISIBLE_DEVICES=1' not in (p/'environ').read_bytes().split(b'\0'):continue
   if any(Path(a).name in ['train.py','train_anchored.py','train_queue.py'] and Path(a).resolve().parent==(S/'scripts').resolve() for a in args if a.endswith('.py')):result.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return result
def own_gpu_pids():
 lines=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 result=[]
 for line in lines.splitlines():
  if not line.strip().isdigit():continue
  try:
   if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():result.append(int(line))
  except FileNotFoundError:pass
 return result
stable=0
while time.time()<DEADLINE:
 live=owned_training();owned=own_gpu_pids()
 free,total=map(int,subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
 stable=stable+1 if not live and not owned and free>=.45*total+2048 else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not owned_training() and not own_gpu_pids()
write(S/'audits/teacher7_expansion_handoff.json',dict(time=time.time(),quiet_checks=stable,free_memory_mib=free,total_memory_mib=total,own_training_pids=live,own_gpu_pids=owned,script_sha256=sha(__file__),scope='No processes signaled. Existing training finishes and its queue exits through STOP_TRAINER. Foreign allocations untouched.'))
command=[sys.executable,str(S/'scripts/generate_targets.py'),'--kind','teacher','--input',str(S/'data/expansion_ready.jsonl'),'--name','teacher7_expansion']
with (S/'logs/generation_teacher7_expansion.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/teacher7_expansion_generation_dispatch.json',dict(time=time.time(),command=command,returncode=r.returncode))
# The child generation function shuts its engine down before returning. Require two quiet checks again.
stable=0
while time.time()<DEADLINE:
 stable=stable+1 if not own_gpu_pids() else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not own_gpu_pids() and not owned_training()
(S/'STOP_TRAINER').unlink()
with (S/'logs/training_after_teacher7_expansion.log').open('w') as log:
 result=subprocess.run([sys.executable,str(S/'scripts/train_queue.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/teacher7_expansion_queue_exit.json',dict(time=time.time(),returncode=result.returncode,generation_returncode=r.returncode))
raise SystemExit(result.returncode)
