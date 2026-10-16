"""Renew verified owned supervisors at an idle stage boundary after user extension."""
from common import *
import fcntl,signal

def main():
    lock=(EXP/'results/stage_execution.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    receipt={'started':time.time(),'action':'User-authorized one-hour extension; renew cached-deadline supervisors between stages'}
    holder=json.loads((EXP/'results/gpu_reservation.json').read_text())
    assert holder['pid']==2393149 and holder['process_start']=='181353007'
    assert json.loads((EXP/'results/gpu_reservation_request.json').read_text())['mode']=='idle'
    for _ in range(30):
        holder=json.loads((EXP/'results/gpu_reservation.json').read_text())
        if holder['mode']=='idle':break
        time.sleep(.5)
    assert holder['mode']=='idle',holder
    targets=[(2393149,'181353007','reserve_gpu.py'),(2393177,'181353207','monitor_gpu.py')]
    verified=[]
    for pid,start,script in targets:
        proc=Path('/proc')/str(pid)
        assert proc.stat().st_uid==os.getuid()
        cmd=(proc/'cmdline').read_bytes().split(b'\0')
        assert str(EXP/'scripts'/script).encode() in cmd
        actual=process_start(pid)
        assert actual and (start is None or start==actual)
        verified.append(dict(pid=pid,process_start=actual,script=script))
    receipt['previous']=verified
    for item in verified:
        assert process_start(item['pid'])==item['process_start']
        os.kill(item['pid'],signal.SIGTERM)
    def gone(pid):
        if process_start(pid) is None:return True
        try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
        except FileNotFoundError:return True
    for _ in range(120):
        if all(gone(x['pid']) for x in verified):break
        time.sleep(.5)
    else:raise RuntimeError('Owned supervisors did not exit')
    for _ in range(60):
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if not any(line.startswith(holder['gpu_uuid']) for line in apps.splitlines()):break
        time.sleep(.5)
    else:raise RuntimeError('GPU occupied; do not touch unverified processes')
    env=dict(os.environ,BENCH_GPU_RESERVER='1',CUDA_VISIBLE_DEVICES='0',BENCH_RUN_ID=EXP.name,BENCH_EXP_DIR=str(EXP),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    launched=[]
    for script in ['reserve_gpu.py','monitor_gpu.py']:
        with (EXP/'logs'/('extended_'+script+'.log')).open('a') as out:
            child=subprocess.Popen([sys.executable,str(EXP/'scripts'/script)],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        launched.append(dict(pid=child.pid,process_start=process_start(child.pid),script=script))
        if script=='reserve_gpu.py':
            for _ in range(120):
                state=json.loads((EXP/'results/gpu_reservation.json').read_text())
                if state.get('pid')==child.pid and state.get('status')=='ready':break
                assert child.poll() is None,'New reservation failed'
                time.sleep(.5)
            else:raise RuntimeError('New reservation not ready')
    receipt.update(finished=time.time(),current=launched,deadline_epoch=1792118298,success=True)
    write(EXP/'provenance/gpu_supervisors_extension_renewal.json',receipt)
    log(receipt)

if __name__=='__main__':
    import sys
    main()
