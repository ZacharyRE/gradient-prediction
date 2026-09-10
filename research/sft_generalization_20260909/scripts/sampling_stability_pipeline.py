"""Serialize GPU3 sampling stability after expansion and new-seed greedy development."""
import subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
from gradient_geometry.sft_protocol import artifact_identity
while time.time()<DEADLINE:
 stage_plan=json.loads((S/'audits/sampling_seed_stability_plan.json').read_text())
 prereqs=[S/'results/generation/teacher32_expansion/complete.json']+[S/'results/evaluation/dev'/n/'math_summary.json' for n in stage_plan['prerequisite_dev_names']]
 if all(p.exists() for p in prereqs):break
 time.sleep(10)
gpu_guard();(S/'STOP_EVALUATOR').touch()
def wait_free():
 while time.time()<DEADLINE:
  n=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if n<1000:return
  time.sleep(10)
 raise RuntimeError('Deadline waiting GPU3')
wait_free();plan=json.loads((S/'audits/sampling_seed_stability_plan.json').read_text());plan['models']={'base':None};plan['adapter_sha256']={};plan['adapter_files_sha256']={}
for seed in [43,44,45]:
 name=f'sample_all_lr1e5_s{seed}_epoch2';path=S/f'results/training/sample_all_lr1e5_s{seed}/epoch2';assert (path/'ready.json').exists();plan['models'][name]=str(path);plan['adapter_sha256'][name]=sha(path/'adapter_model.safetensors');plan['adapter_files_sha256'][name]=artifact_identity(path)['files']
plan['resolved_at']=time.time();plan_path=S/'results/sampling_seed_stability_resolved.json';write(plan_path,plan)
command=[sys.executable,str(S/'scripts/sampling_probe.py'),'--plan',str(plan_path)]
with (S/'logs/sampling_seed_stability.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/sampling_seed_stability_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time()))
wait_free();gpu_guard();(S/'STOP_EVALUATOR').unlink(missing_ok=True)
command=[sys.executable,str(S/'scripts/evaluate.py'),'--queue','--manifest',str(S/'results/protocol_restart_after_sampling.json')]
with (S/'logs/evaluation_after_sampling.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Evaluation ended',r.returncode,flush=True)
