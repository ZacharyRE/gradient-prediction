"""Read-only declared GPU scope and live own-worker allocation audit."""
import argparse,subprocess
from common import *
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
amendment_path=S/'audits/gpu2_additional_execution_amendment.json'
amendment=json.loads(amendment_path.read_text()) if amendment_path.exists() else None
allowed=set(amendment['allowed_physical_gpus']) if amendment else {'1','3'}
commands=read(S/'logs/commands.jsonl');declared=sorted({str(r['gpu']) for r in commands if 'gpu' in r})
assert set(declared)<=allowed,declared
for record in commands:
 if str(record.get('gpu'))=='2':assert amendment and record['time']>=amendment['time']
post_samples=[]
post_path=S/'logs/resources_after_gpu2_authorization.jsonl'
if post_path.exists():
 post_samples=read(post_path)
 assert all(r['scope_passed'] and len(r['observed_study_devices'])<=3 for r in post_samples)
 assert all(r['execution_amendment_sha256']==sha(amendment_path) for r in post_samples)
training=[]
for mp in (S/'results/training').glob('*/manifest.json'):
 m=json.loads(mp.read_text())
 if 'arguments' not in m:continue
 assert m['gpu'] in ['1','3'];training.append(dict(name=mp.parent.name,gpu=m['gpu'],manifest_sha256=sha(mp)))
uuids={}
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines():
 index,uuid=[x.strip() for x in line.split(',')];uuids[uuid]=index
live=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 fields=[x.strip() for x in line.split(',')]
 if len(fields)!=3 or not fields[1].isdigit():continue
 uuid,pid,memory=fields;proc=Path('/proc')/pid
 try:
  if proc.stat().st_uid!=os.getuid():continue
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  env=(proc/'environ').read_bytes().split(b'\0')
  # Print only study workers, never another user's command line/environment.
  scripts=[x for x in argv if x.endswith('.py') and Path(x).resolve().parent==S/'scripts']
  # Renamed vLLM workers must have an actual study-script ancestor.
  ancestor=int(pid);ancestry=[]
  for _ in range(32):
   if ancestor<=1:break
   parent=Path('/proc')/str(ancestor)
   try:
    if parent.stat().st_uid!=os.getuid():break
    av=(parent/'cmdline').read_bytes().decode(errors='replace').split('\0')
    found=[x for x in av if x.endswith('.py') and Path(x).resolve().parent==S/'scripts']
    if found:ancestry.append(dict(pid=ancestor,study_scripts=found));break
    ancestor=int((parent/'stat').read_text().rsplit(')',1)[1].split()[1])
   except (FileNotFoundError,PermissionError,ProcessLookupError):break
  if not ancestry:continue
  gpu=uuids[uuid];assert gpu in allowed,(pid,gpu)
  visible=[e.decode() for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES=')]
  live.append(dict(pid=int(pid),physical_gpu=gpu,gpu_uuid=uuid,memory_mib=memory,study_scripts=scripts,study_ancestry=ancestry,cuda_visible_devices=visible))
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
write(S/f'audits/resource_scope_{args.tag}.json',dict(time=time.time(),deadline=DEADLINE,script_sha256=sha(__file__),commands_sha256=sha(S/'logs/commands.jsonl'),command_records=len(commands),declared_gpus=declared,training=training,live_own_worker_allocations=live,authorized_maximum_simultaneous_gpus=3 if amendment else 2,authorization_sha256=sha(amendment_path) if amendment else None,post_authorization_monitor_sha256=sha(post_path) if post_path.exists() else None,post_authorization_sample_count=len(post_samples),max_post_authorization_observed_gpus=max([len(r['observed_study_devices']) for r in post_samples],default=0),passed=True,scope='All recorded training usedGPUs1/3. Inference commands may additionally useGPU2 only after the explicit user3-GPU authorization. Current own allocations and all available post-authorization process samples respect the amendedmaximum3; samples do not prove continuous historical ownership between polls. Both resource logs containwhole-deviceutilization includingforeignwork and are not this studyFLOPs. No foreign process signaled or private environment recorded.'))
print(json.dumps(dict(passed=True,recorded_command_gpus=declared,training_manifests=len(training),live=live)))
