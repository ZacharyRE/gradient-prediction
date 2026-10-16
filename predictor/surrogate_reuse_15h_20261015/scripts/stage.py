"""Run one study stage on its reserved GPU, with an enforced wall-clock deadline."""
from common import *
import argparse, signal, sys,fcntl

def compute_cutoff_reached():
    return time.time()>=json.loads((EXP/'configs/protocol.json').read_text())['compute_cutoff_epoch']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    assert a.command
    if deadline(900) or compute_cutoff_reached():raise RuntimeError('Compute cutoff before stage admission')
    # FIFO admission prevents a fast queue launcher repeatedly reacquiring the
    # lock ahead of older waiting experiments. Small correctness audits lead.
    waiters=EXP/'results/stage_waiters';waiters.mkdir(exist_ok=True);ticket=waiters/f'{os.getpid()}.json'
    own_start=process_start(os.getpid());priority=0 if a.name.startswith(('audit_','graph_smoke_','replay_graph_smoke_')) else 1
    priorities=EXP/'configs/stage_priorities.json'
    if priority and priorities.exists():
        for prefix,value in json.loads(priorities.read_text())['prefix_priorities'].items():
            if a.name.startswith(prefix):priority=float(value);assert 0<priority<=1
    write(ticket,dict(pid=os.getpid(),process_start=own_start,name=a.name,priority=priority))
    stage_lock=(EXP/'results/stage_execution.lock').open('a+')
    while True:
        if deadline(900) or compute_cutoff_reached():raise RuntimeError('Compute cutoff while waiting for owned stage')
        holder_now=reservation_state();token_now=(holder_now or {}).get('token','');tail=token_now.rsplit('-',1)[-1];running_controller=int(tail) if tail.isdigit() and holder_now.get('mode')=='stage' else None
        pending=[]
        for path in waiters.glob('*.json'):
            try:item=json.loads(path.read_text())
            except FileNotFoundError:continue  # Another controller admitted/exited.
            pid=item['pid'];current=process_start(pid)
            if current!=item['process_start']:
                path.unlink(missing_ok=True);continue
            try:state=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
            except FileNotFoundError:continue
            if state=='Z':path.unlink(missing_ok=True);continue
            if pid!=running_controller:pending.append((item.get('priority',1),int(item['process_start']),pid))
        if pending and min(pending)[2]!=os.getpid():time.sleep(.5);continue
        try:fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if deadline(900) or compute_cutoff_reached():raise RuntimeError('Compute cutoff while waiting for owned stage')
            time.sleep(1)
    ticket.unlink(missing_ok=True)
    cfg=json.loads((EXP/'configs/protocol.json').read_text());token=f'{a.name}-{os.getpid()}'
    if time.time()>=cfg['compute_cutoff_epoch']:raise RuntimeError('Compute cutoff after stage admission')
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
    # Execute immutable source snapshots so concurrent development cannot alter
    # late imports or make a running stage's provenance ambiguous.
    entrypoint=next((Path(s) for s in a.command if s.endswith('.py') and Path(s).parent==EXP/'scripts'),None)
    executed=list(a.command);snapshot=None
    if entrypoint is not None:
        signature=source_signature(entrypoint);snapshot=EXP/'provenance/source_versions'/signature;snapshot.mkdir(parents=True,exist_ok=True)
        sources={p.name:sha(p) for p in sorted((EXP/'scripts').glob('*.py'))}
        for script in (EXP/'scripts').glob('*.py'):
            target=snapshot/script.name
            if not target.exists():shutil.copy2(script,target)
            assert sha(target)==sources[script.name]
        executed=[str(snapshot/Path(s).name) if s.endswith('.py') and Path(s).parent==EXP/'scripts' else s for s in a.command]
        write(EXP/'provenance'/f'stage_source_{a.name}.json',dict(source_signature=signature,sources=sources,original_command=a.command,executed_command=executed,input_config_sha256={s:sha(s) for s in a.command if s.endswith('.json') and Path(s).is_file()}))
    source_dir=snapshot or EXP/'scripts'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',BENCH_RUN_ID=EXP.name,BENCH_EXP_DIR=str(EXP),BENCH_SOURCE_DIR=str(source_dir),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(source_dir)+os.pathsep+os.environ.get('PYTHONPATH',''))
    started=time.time();child=None;reason='finished';last_resource_check=0;verified_owned_contexts={}
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
            child=subprocess.Popen(executed,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
            child_start=process_start(child.pid)
            write(EXP/'results/active_stage.json',dict(name=a.name,pid=child.pid,controller_pid=os.getpid(),process_start=child_start,started=started,command=a.command))
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
                        current_start=process_start(pid)
                        # A verified vLLM/graph child can lose its environ while
                        # NVML still reports its context during teardown. Retain
                        # only its already-verified PID/start identity; a reused
                        # PID with another start time is never admitted here.
                        if pid in verified_owned_contexts and current_start in [verified_owned_contexts[pid],None]:continue
                        try:
                            proc=Path('/proc')/str(pid)
                            own=proc.stat().st_uid==os.getuid() and ('BENCH_RUN_ID='+EXP.name).encode() in (proc/'environ').read_bytes().split(b'\0')
                        except (FileNotFoundError,PermissionError,ProcessLookupError):own=False
                        if own and current_start is not None:verified_owned_contexts[pid]=current_start
                        if not own and not incidental_context_allowed(pid,memory):unexpected.append(line)
                    if unexpected:
                        reason='unexpected_gpu_occupancy'
                        write(EXP/'results'/f'resource_stop_{a.name}.json',dict(time=time.time(),unexpected=unexpected,action='Terminate only this study stage; no foreign process touched'))
                        stop_child()
                        break
                if time.time()>=json.loads((EXP/'configs/protocol.json').read_text())['compute_cutoff_epoch']:
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
