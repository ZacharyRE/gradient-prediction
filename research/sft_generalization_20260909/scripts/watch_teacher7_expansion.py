"""CPU-only provisional build after generation; no training release."""
import subprocess
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
done=S/'results/generation/teacher7_expansion/complete.json'
while time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():
 if done.exists():
  command=[sys.executable,str(S/'scripts/build_teacher7_expansion.py')]
  with (S/'logs/build_teacher7_expansion.log').open('w') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits/teacher7_expansion_build_dispatch.json',dict(time=time.time(),command=command,returncode=result.returncode))
  assert result.returncode==0
  print('Teacher7 expansion provisional build complete;32-case review required.',flush=True)
  break
 time.sleep(20)
