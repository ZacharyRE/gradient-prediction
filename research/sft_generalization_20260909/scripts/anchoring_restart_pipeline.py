"""Resume GPU1 only after readiness and explicit current-trainer/process drain."""
import subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
assert (S/'STOP_TRAINER').exists()
def training_processes():
 found=[];exact={str(S/'scripts'/n) for n in ['train.py','train_anchored.py','train_queue.py']}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv={(x.decode(errors='replace')) for x in (p/'cmdline').read_bytes().split(b'\0')}
   if exact.intersection(argv) and b'CUDA_VISIBLE_DEVICES=1' in (p/'environ').read_bytes().split(b'\0'):found.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return found
stable=0
while time.time()<DEADLINE:
 processes=training_processes();memory=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 stable=stable+1 if (S/'ANCHORING_READY').exists() and not processes and memory<1000 else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not training_processes()
write(S/'audits/anchoring_queue_restart.json',dict(time=time.time(),gpu=1,drained_training_pids=training_processes(),memory_mib=memory,consecutive_quiet_checks=stable,original_train_sha256=sha(S/'scripts/train.py'),queue_script_sha256=sha(S/'scripts/train_queue.py')))
(S/'STOP_TRAINER').unlink();(S/'ANCHORING_READY').unlink()
with (S/'logs/training_after_anchoring.log').open('w') as log:r=subprocess.run([sys.executable,str(S/'scripts/train_queue.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Training queue exited',r.returncode,flush=True)
