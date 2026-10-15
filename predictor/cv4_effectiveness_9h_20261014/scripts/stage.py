"""Run one study stage on its reserved GPU, with an enforced wall-clock deadline."""
from common import *
import argparse, signal, sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    assert a.command
    cfg=json.loads((EXP/'configs/protocol.json').read_text());token=f'{a.name}-{os.getpid()}'
    holder=reservation_state()
    assert holder is not None,'GPU reservation holder is not active'
    occupants=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    contexts=[line for line in occupants.splitlines() if line.split(',')[0].strip()==cfg['gpu_uuid'] and int(line.split(',')[1])!=holder['pid']]
    foreign=[line for line in contexts if not incidental_context_allowed(int(line.split(',')[1]),line.split(',')[2])]
    if foreign:
        rejection=dict(time=time.time(),stage=a.name,foreign_processes=foreign,action='Keep idle reservation; do not release worker VRAM or launch a GPU process')
        with (EXP/'results/admission_rejections.jsonl').open('a') as f:f.write(json.dumps(rejection)+'\n')
        raise RuntimeError('GPU preflight rejected occupied card: '+str(foreign))
    write(EXP/'results/gpu_reservation_request.json',dict(mode='stage',token=token))
    for _ in range(60):
        state=reservation_state()
        if state and state['token']==token:break
        time.sleep(.5)
    else:raise RuntimeError('Reservation handshake failed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',BENCH_RUN_ID=EXP.name,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(EXP/'scripts')+os.pathsep+os.environ.get('PYTHONPATH',''))
    started=time.time();child=None;reason='finished';last_resource_check=0
    def stop_child():
        if child is None or child.poll() is not None:return
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
    try:
        with (EXP/'logs'/f'{a.name}.log').open('a') as out:
            child=subprocess.Popen(a.command,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
            child_start=process_start(child.pid)
            write(EXP/'results/active_stage.json',dict(name=a.name,pid=child.pid,process_start=child_start,started=started,command=a.command))
            while child.poll() is None:
                if time.time()-last_resource_check>=10:
                    last_resource_check=time.time()
                    snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                    unexpected=[]
                    for line in snapshot.splitlines():
                        uuid,pid,memory=[x.strip() for x in line.split(',')];pid=int(pid)
                        if uuid!=cfg['gpu_uuid'] or pid==holder['pid']:continue
                        # During teardown /proc/PID/environ may already be empty
                        # while NVML still lists this exact child context.
                        if pid==child.pid and process_start(pid)==child_start:continue
                        if pid==child.pid and process_start(pid) is None and child.poll() is not None:continue
                        try:
                            proc=Path('/proc')/str(pid)
                            own=proc.stat().st_uid==os.getuid() and ('BENCH_RUN_ID='+EXP.name).encode() in (proc/'environ').read_bytes().split(b'\0')
                        except (FileNotFoundError,PermissionError,ProcessLookupError):own=False
                        if not own and not incidental_context_allowed(pid,memory):unexpected.append(line)
                    if unexpected:
                        reason='unexpected_gpu_occupancy'
                        write(EXP/'results'/f'resource_stop_{a.name}.json',dict(time=time.time(),unexpected=unexpected,action='Terminate only this study stage; no foreign process touched'))
                        stop_child()
                        break
                if time.time()>=cfg['compute_cutoff_epoch']:
                    reason='deadline';stop_child()
                    break
                time.sleep(1)
            code=child.wait()
            if reason!='finished' and code==0:code=1
    finally:
        stop_child()
        write(EXP/'results/gpu_reservation_request.json',dict(mode='idle',token=token+'-done'))
    result=dict(name=a.name,started=started,finished=time.time(),returncode=code,reason=reason,incidental_contexts_at_start=contexts,timing_requires_clean_replay=bool(contexts))
    with (EXP/'results/stage_attempts.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    write(EXP/'results'/f'stage_{a.name}.json',result);log(result)
    sys.exit(code)

if __name__=='__main__':main()
