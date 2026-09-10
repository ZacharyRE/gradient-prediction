"""CPU-only collector for inference restart audit and fixed five-seed development."""
import subprocess
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root=S/'results/evaluation/dev'
tasks=[
 ('restart_replay', ['base_restart_after_sampling','previous_self_sdpa_after_sampling_repeat'],
  S/'audits/sampling_restart_protocol_replay.json', 'audit_sampling_restart.py'),
 ('teacher_five_seed_development',
  [name for seed in [43,44,45,46,47] for name in
   [f'avg_teacher7_1234_s{seed}_epoch4', f'teacher_all_lr5e5_s{seed}_epoch4']],
  S/'audits/teacher_average_five_seed_development.json', 'analyze_teacher_average_extension.py'),
]
done=[]
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 for name,models,output,script in tasks:
  if name in done:continue
  if output.exists():done.append(name);continue
  if name=='teacher_five_seed_development' and 'restart_replay' not in done:continue
  if not all((root/model/'math_summary.json').exists() for model in models):continue
  command=[sys.executable,str(S/'scripts'/script)]
  with (S/'logs'/f'collector_{name}.log').open('w') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits'/f'collector_{name}_dispatch.json',dict(time=time.time(),command=command,returncode=result.returncode))
  assert result.returncode==0 and output.exists(), (name,result.returncode)
  done.append(name);print('Completed',name,flush=True)
 if len(done)==len(tasks):break
 time.sleep(20)
print(json.dumps(dict(done=done,pending=[t[0] for t in tasks if t[0] not in done])),flush=True)
