import subprocess,time
from common import *
while time.time()<DEADLINE and not (S/'STOP_MONITOR').exists():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
 with (S/'logs/resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu=r.stdout,allowed=[1,3]))+'\n')
 time.sleep(30)
