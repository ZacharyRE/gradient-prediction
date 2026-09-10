"""Track whole-device activity and study-owned allocations after the user's3-GPU amendment."""
import subprocess
from common import *
ap=S/'audits/gpu2_additional_execution_amendment.json';a=json.loads(ap.read_text());ah=sha(ap)
assert a['maximum_simultaneous_gpus']==3
uuids={}
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines():
    i,u=[x.strip() for x in line.split(',')];uuids[u]=i
while time.time()<DEADLINE and not (S/'STOP_MONITOR_AUTHORIZED3').exists():
    whole=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    own=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)!=3 or not fields[1].isdigit():continue
        uuid,pid,memory=fields;current=int(pid);found=False
        try:
            if (Path('/proc')/pid).stat().st_uid!=os.getuid():continue
            for _ in range(32):
                if current<=1:break
                p=Path('/proc')/str(current)
                if p.stat().st_uid!=os.getuid():break
                args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
                if any(x.endswith('.py') and Path(x).resolve().parent==S/'scripts' for x in args):found=True;break
                current=int(p.joinpath('stat').read_text().rsplit(')',1)[1].split()[1])
            if found:own.append(dict(pid=int(pid),physical_gpu=uuids[uuid],memory_mib=memory))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    devices=sorted({r['physical_gpu'] for r in own});passed=set(devices)<=set(a['allowed_physical_gpus']) and len(devices)<=3
    record=dict(time=time.time(),gpu=whole,allowed=[1,2,3],maximum_simultaneous_gpus=3,
        study_owned_allocations=own,observed_study_devices=devices,scope_passed=passed,execution_amendment_sha256=ah)
    with (S/'logs/resources_after_gpu2_authorization.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    assert passed,record
    time.sleep(30)
