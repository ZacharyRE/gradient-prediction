"""Record only GPU occupancy; never terminate or modify another process."""
from common import *

def main():
    cfg=json.loads((EXP/'configs/protocol.json').read_text())
    with (EXP/'results/gpu_observations.jsonl').open('a',buffering=1) as out:
        while time.time()<cfg['deadline_epoch']:
            gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm','--format=csv,noheader,nounits'],text=True).strip()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
            rows=[x for x in apps.splitlines() if x.startswith(cfg['gpu_uuid'])]
            out.write(json.dumps(dict(time=time.time(),gpu=gpu,processes=rows))+'\n')
            time.sleep(15)

if __name__=='__main__':main()
