"""Serialize the strong-teacher generation with development evaluation on GPU3."""
import subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
def wait_free():
 while time.time()<DEADLINE:
  r=subprocess.run(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  if int(r.stdout.strip())<1000:return
  time.sleep(10)
 raise RuntimeError('Deadline reached while waiting for GPU3')
wait_free()
command=[sys.executable,str(S/'scripts/generate_strong_targets.py')]
with (S/'logs/generate_teacher32.log').open('w') as log:
 r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/teacher32_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time(),gpu=3))
wait_free();gpu_guard();(S/'STOP_EVALUATOR').unlink(missing_ok=True)
command=[sys.executable,str(S/'scripts/evaluate.py'),'--queue','--manifest',str(S/'results/protocol_restart_after_teacher32.json')]
with (S/'logs/evaluation_after_teacher32.log').open('w') as log:
 r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Evaluation ended',r.returncode,flush=True)
