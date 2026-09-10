"""CPU-only collection of the fixed five-seed validation and resumed-dev audit."""
import subprocess
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
plan=json.loads((S/'audits/teacher_five_seed_math1500_extension_plan.json').read_text())
tasks=[('teacher_five_seed_math1500',[S/'results/evaluation/math_reused1500'/n/'math_summary.json' for n in plan['all_model_identities']],S/'audits/teacher_average_five_seed_math_reused1500.json','analyze_teacher_five_seed_math1500.py'),('teacher1500_restart',[S/'results/evaluation/dev'/n/'math_summary.json' for n in ['base_after_teacher1500_extension','previous_self_sdpa_after_teacher1500_extension']],S/'audits/teacher1500_restart_protocol_replay.json','audit_teacher1500_restart.py')]
done=[]
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 for name,files,output,script in tasks:
  if name in done:continue
  if output.exists():done.append(name);continue
  if not all(p.exists() for p in files):continue
  command=[sys.executable,str(S/'scripts'/script)]
  with (S/'logs'/f'collector_{name}.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits'/f'collector_{name}_dispatch.json',dict(time=time.time(),returncode=r.returncode,command=command))
  assert r.returncode==0 and output.exists();done.append(name);print('Completed',name,flush=True)
 if len(done)==len(tasks):break
 time.sleep(20)
