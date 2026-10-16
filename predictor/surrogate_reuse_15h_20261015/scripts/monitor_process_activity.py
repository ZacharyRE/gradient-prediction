"""Read NVML per-process utilization samples; never modify GPU processes."""
from common import *
import pynvml as nv

def main():
    cfg=json.loads((EXP/'configs/protocol.json').read_text());nv.nvmlInit()
    handle=nv.nvmlDeviceGetHandleByIndex(0);last=int(time.time()*1e6)
    with (EXP/'results/gpu_process_activity.jsonl').open('a',buffering=1) as out:
        while time.time()<cfg['deadline_epoch']:
            try:
                samples=nv.nvmlDeviceGetProcessUtilization(handle,last)
                rows=[dict(pid=s.pid,time_us=s.timeStamp,sm_percent=s.smUtil,memory_percent=s.memUtil) for s in samples]
                if rows:last=max(r['time_us'] for r in rows)
                status='samples' if rows else 'no_new_samples'
            except nv.NVMLError as e:rows=[];status=type(e).__name__
            record=dict(time=time.time(),status=status,process_samples=rows,total_gpu_percent=nv.nvmlDeviceGetUtilizationRates(handle).gpu)
            out.write(json.dumps(record)+'\n')
            time.sleep(1)
    nv.nvmlShutdown()

if __name__=='__main__':main()
