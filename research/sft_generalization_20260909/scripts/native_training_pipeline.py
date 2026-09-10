"""After a clean trainer queue stop, run native diagnostics then resume training."""
import argparse,subprocess
from common import *
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--judge-input',type=Path);a=p.parse_args();gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
pidfile=Path(f'/proc/{a.wait_pid}/stat')
initial=pidfile.read_text().rsplit(')',1)[1].split()[19] if pidfile.exists() else None
while time.time()<DEADLINE:
 try:alive=initial is not None and pidfile.read_text().rsplit(')',1)[1].split()[19]==initial
 except FileNotFoundError:alive=False
 if not alive:break
 time.sleep(10)
def wait_free():
 while time.time()<DEADLINE:
  r=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  if int(r.stdout.strip())<1000:return
  time.sleep(10)
 raise RuntimeError('Deadline reached waiting for GPU1')
wait_free();gpu_guard()
if a.judge_input:
 command=[sys.executable,str(S/'scripts/judge_targets.py'),'--input',str(a.judge_input),'--name','guided_legacy']
 with (S/'logs/judge_guided.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
 write(S/'audits/judge_guided_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time(),gpu=1))
 wait_free();gpu_guard()
command=[sys.executable,str(S/'scripts/native_recheck.py'),'--manifest',str(S/'results/native_recheck_manifest.json')]
with (S/'logs/native_recheck.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/native_recheck_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time(),gpu=1,waited_for_pid=a.wait_pid,waited_for_start=initial))
wait_free();gpu_guard()
tasks=S/'results/gpu1_pretrain_tasks.json'
if tasks.exists():
 for job in json.loads(tasks.read_text()):
  assert job['script'] in ['score_candidates.py']
  command=[sys.executable,str(S/'scripts'/job['script']),*job['arguments']]
  with (S/'logs'/job['log']).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits'/job['audit'],dict(command=command,returncode=r.returncode,time=time.time(),gpu=1))
  wait_free();gpu_guard()
(S/'STOP_TRAINER').unlink(missing_ok=True)
with (S/'logs/training_after_native.log').open('w') as log:r=subprocess.run([sys.executable,str(S/'scripts/train_queue.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Training queue ended',r.returncode,flush=True)
