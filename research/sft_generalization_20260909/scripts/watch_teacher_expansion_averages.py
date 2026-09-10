"""Derive each prospectively specified scale average as its sources become ready."""
import subprocess
from common import *
plan=json.loads((S/'audits/teacher_followup_controls_plan.json').read_text());jobs=plan['derived_jobs'];done=[]
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 available=[j for j in jobs if j['name'] not in done and all((Path(p)/'ready.json').exists() for p in j['sources'])]
 if available:
  assert sha(S/'scripts/average_epoch_adapters.py')==plan['averager_sha256']
  subprocess.run([sys.executable,str(S/'scripts/average_epoch_adapters.py')],check=True)
  for j in available:
   dest=S/'results/training'/j['name']/f"epoch{j['endpoint']}"
   assert (dest/'ready.json').exists();done.append(j['name'])
  write(S/'audits/teacher_expansion_average_dispatch.json',dict(time=time.time(),plan_sha256=sha(S/'audits/teacher_followup_controls_plan.json'),completed=done,all_complete=len(done)==len(jobs)))
 if len(done)==len(jobs):break
 time.sleep(15)
