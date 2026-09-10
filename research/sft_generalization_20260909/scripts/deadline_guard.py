"""Enforce the authorized wall-time bound, targeting this study's GPU worker trees."""
import signal
from common import *
WORKERS={'wait_and_run.py','train_queue.py','train.py','evaluate.py','generate_targets.py','judge_targets.py','teacher32_pipeline.py','generate_strong_targets.py'}
tracked={}
def processes():
 result={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   stat=(p/'stat').read_text().rsplit(')',1)[1].split()
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   result[int(p.name)]=dict(start=stat[19],ppid=int(stat[1]),args=args)
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return result
while not (S/'STOP_DEADLINE_GUARD').exists():
 registry=S/'scripts/gpu_worker_registry.json'
 workers=WORKERS|set(json.loads(registry.read_text())) if registry.exists() else WORKERS
 procs=processes();owned=set()
 for pid,r in procs.items():
  for arg in r['args'][1:]:
   if arg.endswith('.py'):
    p=Path(arg)
    if p.name in workers and p.resolve().parent==(S/'scripts').resolve():owned.add(pid)
 changed=True
 while changed:
  children={pid for pid,r in procs.items() if r['ppid'] in owned};changed=bool(children-owned);owned.update(children)
 for pid in owned:tracked[pid]=procs[pid]
 if time.time()>=DEADLINE-30:
  for name in ['STOP_TRAINER','STOP_EVALUATOR','STOP_MONITOR']:(S/name).touch()
  targeted=[]
  for sig in [signal.SIGTERM,signal.SIGKILL]:
   live=processes()
   for pid,r in tracked.items():
    if pid in live and live[pid]['start']==r['start']:
     try:os.kill(pid,sig);targeted.append(dict(pid=pid,signal=int(sig),args=r['args']))
     except ProcessLookupError:pass
   if sig==signal.SIGTERM:time.sleep(5)
  write(S/'audits/deadline_enforcement.json',dict(time=time.time(),deadline=DEADLINE,targeted=targeted,method='Study GPU worker roots plus descendants; PID start times checked to avoid PID reuse; 30-second deadline buffer'))
  break
 time.sleep(min(20,max(1,DEADLINE-30-time.time())))
