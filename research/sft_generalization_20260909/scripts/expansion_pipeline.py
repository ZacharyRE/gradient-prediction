"""Drain current13 trained models' dev results, expand targets, resume same evaluator."""
import subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
plan=json.loads((S/'audits/expansion_generation_plan.json').read_text());last=None
while time.time()<DEADLINE:
 missing=[n for n in plan['wait_for_development'] if not (S/'results/evaluation/dev'/n/'math_summary.json').exists()]
 if not missing:break
 if len(missing)!=last:print('Waiting development checkpoints',len(missing),flush=True);last=len(missing)
 time.sleep(10)
gpu_guard();assert (S/'audits/expansion_release_for_generation.json').exists()
(S/'STOP_EVALUATOR').touch()
def wait_free():
 while time.time()<DEADLINE:
  used=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
  if int(used)<1000:return
  time.sleep(10)
 raise RuntimeError('Deadline reached waiting for GPU3')
wait_free();gpu_guard()
command=[sys.executable,str(S/'scripts/generate_strong_targets.py'),'--input',str(S/'data/expansion_ready.jsonl'),'--name','teacher32_expansion']
with (S/'logs/generate_expansion.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/expansion_generation_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time(),gpu=3))
wait_free();gpu_guard();(S/'STOP_EVALUATOR').unlink(missing_ok=True)
command=[sys.executable,str(S/'scripts/evaluate.py'),'--queue','--manifest',str(S/'results/protocol_restart_after_expansion.json')]
with (S/'logs/evaluation_after_expansion.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Evaluation ended',r.returncode,flush=True)
