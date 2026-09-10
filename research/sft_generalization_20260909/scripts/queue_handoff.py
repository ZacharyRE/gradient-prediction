"""One-time replacement of study CPU dispatcher; active training child is left running."""
import psutil,signal,subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1';plan=json.loads((S/'audits/queue_handoff_plan.json').read_text());parent=psutil.Process(plan['parent_pid']);assert parent.create_time()==plan['parent_create_time'];assert parent.cmdline()[1]==str(S/'scripts/train_queue.py')
# Freeze only dispatcher's Python process to avoid launching a second child during handoff.
parent.send_signal(signal.SIGSTOP);children=parent.children(recursive=False)
assert len(children)<=1
for child in children:assert child.cmdline()[1]==str(S/'scripts/train.py')
tracked=[dict(pid=c.pid,create_time=c.create_time(),command=c.cmdline()) for c in children];write(S/'audits/queue_handoff_started.json',dict(time=time.time(),parent=plan,children=tracked,policy='Only CPU dispatcher receives signals; active training child completes unchanged.'))
parent.send_signal(signal.SIGTERM);parent.send_signal(signal.SIGCONT)
while time.time()<DEADLINE:
 alive=[]
 for c in tracked:
  try:
   p=psutil.Process(c['pid']);alive.append(p.create_time()==c['create_time'] and p.status()!=psutil.STATUS_ZOMBIE)
  except psutil.NoSuchProcess:alive.append(False)
 if not any(alive):break
 time.sleep(5)
while time.time()<DEADLINE:
 used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 if used<1000:break
 time.sleep(5)
gpu_guard();assert not (S/'STOP_TRAINER').exists();write(S/'audits/queue_handoff_ready.json',dict(time=time.time(),gpu=1,children_completed=tracked))
with (S/'logs/training_dynamic.log').open('w') as log:r=subprocess.run([sys.executable,str(S/'scripts/train_queue.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Dynamic training dispatcher ended',r.returncode,flush=True)
