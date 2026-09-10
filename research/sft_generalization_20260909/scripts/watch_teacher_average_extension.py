"""CPU-only derivation after both already-authorized replication runs complete."""
import subprocess
from common import *
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
plan=json.loads((S/'audits/teacher_average_seed_extension_plan.json').read_text())
requirements=[S/'results/training'/j['name']/'complete.json' for j in plan['training_jobs']]
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 if all(p.exists() for p in requirements):
  command=[sys.executable,str(S/'scripts/average_epoch_adapters.py')]
  with (S/'logs/teacher_average_seed_extension_derivation.log').open('w') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=max(1,DEADLINE-time.time()))
  ready=[(S/'results/training'/j['name']/f"epoch{j['endpoint']}"/'ready.json').exists() for j in plan['derived_jobs']]
  write(S/'audits/teacher_average_seed_extension_derivation.json',dict(time=time.time(),returncode=result.returncode,ready=ready,command=command,plan_sha256=sha(S/'audits/teacher_average_seed_extension_plan.json')))
  assert result.returncode==0 and all(ready);break
 time.sleep(10)
