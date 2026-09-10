"""CPU-only collection of already frozen diagnostics; never releases training data."""
import subprocess
from common import *

assert not os.environ.get('CUDA_VISIBLE_DEVICES')
train_plan=json.loads((S/'audits/train_free_probe_plan.json').read_text())
reuse_plan=json.loads((S/'results/math_reused1500_manifest.json').read_text())
jobs=[
 ('long_budget_analysis',[S/'results/evaluation_long/dev/complete.json'],S/'results/long_budget_analysis.json',['analyze_long_budget.py']),
 ('seen_training_generation_analysis',[S/'results/evaluation/train_free_probe256'/n/'math_summary.json' for n in train_plan['models']],S/'results/train_free_generation_analysis.json',['analyze_train_free.py']),
 ('sample_expansion_provisional_build',[S/'results/generation/sample_expansion/complete.json',S/'results/evaluation/expansion_ready/base/math_summary.json'],S/'audits/sample_expansion_build.json',['build_sample_expansion.py']),
 ('fixed_families_reused1500',[S/'results/evaluation/math_reused1500'/n/'math_summary.json' for n in reuse_plan],S/'audits/fixed_seed_math_reused1500.json',['analyze_fixed_seed_families.py','--dataset','math_reused1500'])]
done=set()
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 for name,requirements,output,args in jobs:
  if name in done:continue
  output_complete=output.exists()
  if output_complete and name=='fixed_families_reused1500':
   families=json.loads(output.read_text()).get('families',{})
   output_complete=len(families)==4 and all(not f.get('missing') and len(f.get('per_seed',[]))==3 for f in families.values())
  if output_complete:done.add(name);continue
  if not all(p.exists() for p in requirements):continue
  command=[sys.executable,str(S/'scripts'/args[0]),*args[1:]]
  with (S/'logs'/('collect_'+name+'.log')).open('w') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  record=dict(time=time.time(),name=name,command=command,returncode=result.returncode,output=str(output),output_exists=output.exists(),cpu_only=True)
  write(S/'audits'/('collect_'+name+'.json'),record);print(json.dumps(record),flush=True);done.add(name)
 if len(done)==len(jobs):break
 time.sleep(30)
print(json.dumps(dict(finished=time.time(),collected=sorted(done))),flush=True)
