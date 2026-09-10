"""Reload priority queue after each complete training job; never interrupt a training child."""
import subprocess
from common import *
gpu_guard()
while time.time()<DEADLINE:
 if (S/'STOP_TRAINER').exists():break
 jobs=json.loads((S/'results/train_queue.json').read_text());job=None
 for candidate in jobs:
  out=S/'results/training'/candidate['name']
  if (out/'complete.json').exists() or (S/'logs'/('failed_'+candidate['name']+'.json')).exists():continue
  if not Path(candidate['train_file']).exists():continue
  job=candidate;break
 if job is None:time.sleep(10);continue
 trainer=job.get('trainer_script','train.py');assert trainer in ['train.py','train_anchored.py']
 command=[sys.executable,str(S/'scripts'/trainer),'--name',job['name'],'--train-file',job['train_file']]
 for key,value in job.items():
  if key not in ['name','train_file','trainer_script']:command+=['--'+key.replace('_','-'),str(value)]
 with (S/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],command=command))+'\n')
 with (S/'logs'/('train_'+job['name']+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
 print(json.dumps(dict(name=job['name'],returncode=r.returncode,time=time.time())),flush=True)
 if r.returncode:write(S/'logs'/('failed_'+job['name']+'.json'),dict(returncode=r.returncode,time=time.time()))
