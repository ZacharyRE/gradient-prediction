"""Bounded four-seed replication, with natural GPU1 handoff and no queue resume."""
import subprocess,datetime
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
pp=S/'audits/teacher7_expansion_seed_replication_plan.json';plan=json.loads(pp.read_text());ph=sha(pp)
start_limit=datetime.datetime(2026,9,10,6,tzinfo=datetime.timezone.utc).timestamp()
finish_limit=datetime.datetime(2026,9,10,7,tzinfo=datetime.timezone.utc).timestamp()
sentinel=S/'STOP_TRAINER'
def live_workers():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   if b'CUDA_VISIBLE_DEVICES=1' not in (p/'environ').read_bytes().split(b'\0'):continue
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if any(Path(x).name in {'train.py','train_anchored.py','train_queue.py'} and Path(x).resolve().parent==S/'scripts' for x in args if x.endswith('.py')):result.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return result
def gpu_state():
 owned=[]
 for line in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
  if not line.strip().isdigit():continue
  try:
   if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():owned.append(int(line))
  except FileNotFoundError:pass
 free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 return owned,free
checks=[];stable=0
while time.time()<min(DEADLINE,start_limit):
 done=(S/'results/training'/plan['wait_for_training']/'complete.json').exists()
 workers=live_workers()
 if done and sentinel.exists() and not workers:
  owned,free=gpu_state();good=not owned and free>=45*1024
  checks.append(dict(time=time.time(),owned_gpu_pids=owned,free_mib=free,workers=workers,quiet=good));stable=stable+1 if good else 0
  if stable>=2:break
 time.sleep(5)
else:raise RuntimeError('Replication handoff start cutoff reached')
write(S/'audits/expanded_seed_replication_handoff.json',dict(time=time.time(),gpu='1',plan_sha256=ph,checks=checks,script_sha256=sha(__file__),scope='Old queue and trainers finished naturally; no process signaled. STOP_TRAINER remains set throughout explicit four-job replication.'))
dispatch=[]
for job in plan['jobs']:
 gpu_guard();assert sentinel.exists() and sha(pp)==ph
 assert sha(S/'scripts/train.py')==plan['trainer_sha256'] and sha(job['train_file'])==plan['data_sha256']
 assert not (S/'results/training'/job['name']).exists()
 command=[sys.executable,str(S/'scripts/train.py'),'--name',job['name'],'--train-file',job['train_file']]
 for key,value in job.items():
  if key not in ['name','train_file']:command+=['--'+key.replace('_','-'),str(value)]
 with (S/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu='1',command=command))+'\n')
 with (S/f"logs/train_{job['name']}.log").open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,min(DEADLINE,finish_limit)-time.time()))
 dispatch.append(dict(time=time.time(),name=job['name'],returncode=r.returncode,command=command));write(S/'audits/expanded_seed_replication_dispatch.json',dict(plan_sha256=ph,tasks=dispatch,all_complete=False))
 if r.returncode:raise SystemExit(r.returncode)
 assert (S/'results/training'/job['name']/'complete.json').exists()
 assert sha(S/'scripts/average_epoch_adapters.py')==plan['averager_sha256']
 subprocess.run([sys.executable,str(S/'scripts/average_epoch_adapters.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
for job in plan['derived_jobs']:assert (S/'results/training'/job['name']/f"epoch{job['endpoint']}"/'ready.json').exists()
write(S/'audits/expanded_seed_replication_dispatch.json',dict(time=time.time(),plan_sha256=ph,tasks=dispatch,all_complete=True))
print('All four expanded-teacher seeds and averages complete; GPU1 remains reserved.',flush=True)
