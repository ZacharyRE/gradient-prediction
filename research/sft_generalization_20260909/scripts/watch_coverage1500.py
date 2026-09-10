"""CPU-only fixed-pilot analysis and post-detour protocol replay."""
import subprocess
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
plan=json.loads((S/'audits/coverage_math1500_plan.json').read_text())
tasks=[('coverage_math1500',[S/'results/evaluation/math_reused1500'/n/'math_summary.json' for n in plan['models']],S/'audits/coverage_math1500_analysis.json','analyze_coverage_math1500.py'),('coverage1500_restart',[S/'results/evaluation/dev'/n/'math_summary.json' for n in ['base_after_coverage1500','previous_self_sdpa_after_coverage1500']],S/'audits/coverage1500_restart_protocol_replay.json','audit_coverage1500_restart.py')]
done=[]
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 for name,files,output,script in tasks:
  if name in done:continue
  if output.exists():done.append(name);continue
  if not all(p.exists() for p in files):continue
  command=[sys.executable,str(S/'scripts'/script)]
  with (S/'logs'/f'collector_{name}.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits'/f'collector_{name}_dispatch.json',dict(time=time.time(),returncode=r.returncode,command=command))
  assert r.returncode==0 and output.exists();done.append(name)
 if len(done)==len(tasks):break
 time.sleep(20)
