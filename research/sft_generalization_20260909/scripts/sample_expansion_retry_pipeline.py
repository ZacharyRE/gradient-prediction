"""Retry isolated generation after explicit CPU/GPU process drain; preserve incident artifacts."""
import subprocess
from common import *
gpu_guard();assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
(S/'STOP_TRAINER').touch()

def training_processes():
 found=[]
 exact={str(S/'scripts/train.py'),str(S/'scripts/train_queue.py')}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if not exact.intersection(x.decode(errors='replace') for x in argv):continue
   env=(p/'environ').read_bytes().split(b'\0')
   if b'CUDA_VISIBLE_DEVICES=1' in env:found.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return found

def wait_quiet():
 stable=0
 while time.time()<DEADLINE:
  processes=training_processes();memory=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  stable=stable+1 if not processes and memory<1000 else 0
  if stable>=2:return
  time.sleep(5)
 raise RuntimeError('Deadline waiting for actual training-process drain')

wait_quiet();gpu_guard()
original=S/'results/generation/sample_expansion';partial=original.with_name('sample_expansion_concurrent_partial')
assert original.exists() and not partial.exists();original.rename(partial)
log=S/'logs/generate_sample_expansion.log';assert log.exists();log.rename(log.with_name('generate_sample_expansion_concurrent_partial.log'))
write(S/'audits/sample_expansion_retry_started.json',dict(time=time.time(),drained_training_pids=training_processes(),partial_generation=str(partial),partial_manifest_sha256=sha(partial/'manifest.json'),reason='Explicit process-drain andtwoquietGPUchecks, not memoryalone. Partial generation retained solely for numerical comparison; fresh isolated full generation used for targets.'))
command=[sys.executable,str(S/'scripts/generate_targets.py'),'--kind','sample','--input',str(S/'data/expansion_ready.jsonl'),'--name','sample_expansion']
with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
write(S/'audits/sample_expansion_generation_dispatch.json',dict(time=time.time(),command=command,returncode=r.returncode,gpu=1,isolated_retry=True))
assert r.returncode==0
old=read(partial/'predictions.jsonl');new=read(original/'predictions.jsonl');lookup={(r['index'],r['sample']):r for r in new}
equal={k:sum(r[k]==lookup[(r['index'],r['sample'])][k] for r in old) for k in ['prediction','correct','finish_reason','generated_tokens']}
write(S/'audits/sample_expansion_concurrency_replay.json',dict(time=time.time(),partial_rows=len(old),equal_counts=equal,partial_sha256=sha(partial/'predictions.jsonl'),isolated_sha256=sha(original/'predictions.jsonl'),interpretation='Only the isolated generation is released for training. Prefix equality, if observed, checks measured numerical behavior; no blanket equivalence claim for arbitrary concurrent kernels.'))
wait_quiet();gpu_guard();(S/'STOP_TRAINER').unlink(missing_ok=True)
with (S/'logs/training_after_sample_expansion.log').open('w') as f:r=subprocess.run([sys.executable,str(S/'scripts/train_queue.py')],stdout=f,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
print('Training queue ended',r.returncode,flush=True)
