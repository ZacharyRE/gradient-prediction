"""Bounded six-model validation detour, with explicit GPU3 ownership drain."""
import subprocess
from common import *
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
plan=json.loads((S/'audits/coverage_math1500_plan.json').read_text())
def active_queue():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if '--queue' not in args or b'CUDA_VISIBLE_DEVICES=3' not in (p/'environ').read_bytes().split(b'\0'):continue
   if any(Path(a).resolve()==(S/'scripts/evaluate.py').resolve() for a in args if a.endswith('.py')):result.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return result
def own_gpu_pids():
 result=[]
 for line in subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
  if not line.strip().isdigit():continue
  try:
   if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():result.append(int(line))
  except FileNotFoundError:pass
 return result
while time.time()<DEADLINE:
 if all((S/'results/evaluation/dev'/name/'math_summary.json').exists() for name in plan['wait_for_dev_primary']):break
 time.sleep(10)
gpu_guard()
from gradient_geometry.sft_protocol import artifact_identity
ready={json.loads(p.read_text())['name']:Path(json.loads(p.read_text())['adapter']) for p in (S/'results/training').glob('*/epoch*/ready.json')}
identities={};manifest={}
for name,spec in plan['models'].items():
 path=ready[name];assert not (S/'results/evaluation/math_reused1500'/name/'math_predictions.jsonl').exists()
 train=json.loads((S/'results/training'/spec['run']/'manifest.json').read_text())
 for key,value in plan['expected_arguments'].items():assert train['arguments'][key]==value,(name,key)
 assert train['data_sha']==plan['training_data_sha256'][spec['dataset']]
 assert train['script_sha']==sha(S/'scripts/train.py') and train['arguments'].get('resume_from') is None
 assert sha(Path(train['arguments']['train_file']))==train['data_sha']
 if spec['kind']=='average':
  deriv=json.loads((path/'derivation.json').read_text());assert deriv['script_sha256']==plan['averager_sha256'] and len(deriv['sources'])==4
  for e,item in enumerate(deriv['sources'],1):
   expected=S/'results/training'/spec['run']/f'epoch{e}'
   assert Path(item['path']).resolve()==expected.resolve() and sha(expected/'adapter_model.safetensors')==item['weight_sha256']
   assert sha(expected/'adapter_config.json')==item['config_sha256']
 manifest[name]=str(path);identities[name]=artifact_identity(path)
assert sha(S/'data/math_reused1500.jsonl')==plan['data_sha256']
write(S/'results/coverage_math1500_manifest.json',manifest)
base=json.loads((S/'results/evaluation/math_reused1500/base/math_manifest.json').read_text())
assert not (S/'audits/coverage_math1500_frozen.json').exists(), 'Do not overwrite frozen pilot identities'
write(S/'audits/coverage_math1500_frozen.json',dict(time=time.time(),prospective_plan_sha256=sha(S/'audits/coverage_math1500_plan.json'),model_identities=identities,manifest_sha256=sha(S/'results/coverage_math1500_manifest.json'),data_sha256=plan['data_sha256'],protocol={k:base[k] for k in ['model','engine','batch_invariant','max_tokens','temperature','chunk_size','script_sha256']},scope='All six exact identities frozen before any of their reused1500 outputs. Single-seed development followup only.'))
(S/'STOP_EVALUATOR').touch();stable=0
while time.time()<DEADLINE:
 free,total=map(int,subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
 stable=stable+1 if not active_queue() and not own_gpu_pids() and free>=.30*total+2048 else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not active_queue() and not own_gpu_pids()
write(S/'audits/coverage1500_handoff.json',dict(time=time.time(),quiet_checks=stable,own_gpu_pids=[],free_memory_mib=free,script_sha256=sha(__file__),scope='No process signaled; existing dev queue finished its current model and exited. Foreign allocations untouched.'))
command=[sys.executable,str(S/'scripts/evaluate.py'),'--dataset','math_reused1500','--manifest',str(S/'results/coverage_math1500_manifest.json')]
with (S/'logs/evaluation_coverage1500.log').open('w') as log:
 r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/coverage1500_dispatch.json',dict(time=time.time(),returncode=r.returncode,command=command))
if r.returncode:raise SystemExit(r.returncode)
stable=0
while time.time()<DEADLINE:
 stable=stable+1 if not own_gpu_pids() else 0
 if stable>=2:break
 time.sleep(5)
gpu_guard();assert not own_gpu_pids() and not active_queue();(S/'STOP_EVALUATOR').unlink()
command=[sys.executable,str(S/'scripts/evaluate.py'),'--queue','--manifest',str(S/'results/protocol_restart_after_coverage1500.json')]
with (S/'logs/evaluation_after_coverage1500.log').open('w') as log:
 result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/coverage1500_queue_exit.json',dict(time=time.time(),returncode=result.returncode,extension_returncode=r.returncode))
raise SystemExit(result.returncode or r.returncode)
