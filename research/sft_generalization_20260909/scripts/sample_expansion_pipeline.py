"""After fixed teacher7 seedreplicas, mine new-question student targets onGPU1, resume queue."""
import subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
while time.time()<DEADLINE:
 if all((S/f'results/training/teacher_all_lr5e5_s{seed}/complete.json').exists() for seed in [44,45]):break
 time.sleep(2)
gpu_guard();(S/'STOP_TRAINER').touch()
def wait_free():
 while time.time()<DEADLINE:
  n=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if n<1000:return
  time.sleep(5)
 raise RuntimeError('Deadline waitingGPU1')
wait_free();gpu_guard();command=[sys.executable,str(S/'scripts/generate_targets.py'),'--kind','sample','--input',str(S/'data/expansion_ready.jsonl'),'--name','sample_expansion']
with (S/'logs/generate_sample_expansion.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/sample_expansion_generation_dispatch.json',dict(time=time.time(),command=command,returncode=r.returncode,gpu=1))
wait_free();gpu_guard();(S/'STOP_TRAINER').unlink(missing_ok=True)
with (S/'logs/training_after_sample_expansion.log').open('w') as log:r=subprocess.run([sys.executable,str(S/'scripts/train_queue.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Training queue ended',r.returncode,flush=True)
