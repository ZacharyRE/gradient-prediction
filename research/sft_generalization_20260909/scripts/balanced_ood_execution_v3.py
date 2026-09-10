"""Balance frozen main OOD first, then optional supplementary evaluations on GPU2/3."""
import signal
import subprocess
from common import *

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ap = S/'audits/balanced_ood_zombie_read_amendment.json'
amend = json.loads(ap.read_text()); ah = sha(ap)
assert amend['execution_script_sha256'] == sha(__file__)
assert sha(S/'results/confirmation_plan.json') == amend['primary_plan_sha256']
assert sha(S/'audits/expanded_unaveraged_supplement_plan.json') == amend['supplement_plan_sha256']
for filename, digest in amend['unchanged_scripts'].items():
    assert sha(S/'scripts'/filename) == digest
plan = json.loads((S/'results/confirmation_plan.json').read_text())
supp = json.loads((S/'audits/expanded_unaveraged_supplement_plan.json').read_text())
finish = amend['ood_finish_cutoff']
os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', '')
audit = S/'audits/balanced_ood_execution.json'
resume_path=S/amend['resume_state_path']
assert sha(resume_path)==amend['resume_state_sha256']
resume=json.loads(resume_path.read_text())
assert not resume['running'] and not resume['calls']
assert all(t['status']=='pending' for t in resume['tasks'])
tasks = [dict(x, status='pending') for x in amend['tasks']]
parents = resume['parents']
assert parents['3']['phase']=='handoff_child_running_parent_paused'
assert parents['2']['phase']=='handoff_child_running_parent_paused'
parents['2'].update(amend['parents']['2'])
running = {}; logs = {}; calls = []; errors = []
main_names = list(json.loads((S/'results/confirmation_manifest.json').read_text()))
supp_names = list(supp['models'].values())

def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        if p.stat().st_uid != os.getuid():
            return None
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            # An exited child has no live environment; /proc/environ can deny access.
            # Preserve UID, PID birth, parent and kernel exit status for handoff.
            return dict(pid=pid, start=fields[19], state=fields[0], ppid=int(fields[1]),
                        exit_code=int(fields[49]), args=[], env=[], children=[])
        return dict(pid=pid, start=fields[19], state=fields[0], ppid=int(fields[1]),
                    exit_code=int(fields[49]), args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0'),
                    env=(p/'environ').read_bytes().split(b'\0'),
                    children=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None

def identity(r):
    return {k:r[k] for k in ['pid','start','state','ppid','exit_code','args']}

def same_process(expected):
    r = proc(expected['pid'])
    return r if r and r['start'] == expected['start'] else None

def summaries_complete(dataset, names):
    for name in names:
        path = S/'results/evaluation'/dataset/name/'math_summary.json'
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        if record['samples'] != amend['dataset_sizes'][dataset]:
            raise RuntimeError((dataset, name, 'wrong sample count'))
    return True

def gpu_quiet(gpu):
    own = []
    raw = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        if line.strip().isdigit():
            p = Path('/proc')/line.strip()
            try:
                if p.stat().st_uid == os.getuid():
                    own.append(int(line))
            except FileNotFoundError:
                pass
    free, total = map(int, subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'], text=True).strip().split(','))
    return dict(time=time.time(), own_gpu_pids=own, free_mib=free,
                quiet=not own and free >= max(45*1024, .30*total+2048))

def publish(status):
    main_complete = parents['3']['phase']=='available' and summaries_complete('math_reused5000', main_names) and all(t['status']=='complete' for t in tasks if t['role']=='main')
    supp_complete = parents['2']['phase']=='available' and summaries_complete('math_reused5000', supp_names) and all(t['status']=='complete' for t in tasks if t['role']=='supplement')
    write(audit, dict(time=time.time(), status=status, all_complete=status=='all_gpu_evaluations_complete',
        main_gpu_complete=main_complete, supplement_gpu_complete=supp_complete,
        execution_amendment_sha256=ah, script_sha256=sha(__file__), parents=parents,
        tasks=tasks, running={g:dict(pid=p.pid,task_id=t['id']) for g,(p,t) in running.items()},
        calls=calls, errors=errors, scope=amend['scope']))
    # Native-control status belongs to the unchanged GPU2 native runner and is untouched.
    # Supplementary dispatch ownership transfers when the replay parent is paused.
    if parents['2']['phase'] != 'waiting_handoff_child':
        terminal = status in ['all_gpu_evaluations_complete','incomplete_error_or_cutoff','error']
        value = 'all_gpu_evaluations_complete' if supp_complete else ('incomplete_error_or_cutoff' if terminal else 'running_balanced_ood_execution')
        old = json.loads((S/'audits/expanded_unaveraged_supplement_status.json').read_text()) if (S/'audits/expanded_unaveraged_supplement_status.json').exists() else {}
        write(S/'audits/expanded_unaveraged_supplement_status.json', dict(time=time.time(), status=value,
            all_complete=supp_complete, plan_sha256=amend['supplement_plan_sha256'],
            original_gpu2_execution_amendment_sha256=amend['gpu2_execution_amendment_sha256'],
            balanced_execution_amendment_sha256=ah,
            actual_gpus=sorted({'2'} | {t['gpu'] for t in tasks if t['role']=='supplement' and 'gpu' in t}),
            prior_gpu2_calls=old.get('prior_gpu2_calls', old.get('calls', [])),
            balanced_task_ids=[t['id'] for t in tasks if t['role']=='supplement']))

def capture_math_parent(gpu):
    state = parents[gpu]
    r = same_process(state)
    if not r:
        raise RuntimeError((gpu, 'planned CPU parent disappeared before handoff'))
    assert ('CUDA_VISIBLE_DEVICES='+gpu).encode() in r['env']
    assert any(Path(x).resolve()==S/'scripts'/state['script'] for x in r['args'] if x.endswith('.py'))
    match = []
    for pid in r['children']:
        child = proc(pid)
        if not child:
            continue
        args = child['args']
        if '--dataset' not in args or args[args.index('--dataset')+1] != state['child_dataset']:
            continue
        if '--manifest' not in args or Path(args[args.index('--manifest')+1]).resolve() != Path(state['manifest']):
            continue
        if not any(Path(x).resolve()==S/'scripts/evaluate.py' for x in args if x.endswith('.py')):
            continue
        assert ('CUDA_VISIBLE_DEVICES='+gpu).encode() in child['env']
        match.append(child)
    if not match:
        # GPU2 is intentionally left running through native controls until the vLLM replay child starts.
        if gpu == '2':
            return
        raise RuntimeError('GPU3 expected current MATH child not found')
    assert len(match) == 1
    if gpu == '2':
        assert json.loads((S/'audits/native_target_control_status.json').read_text())['all_complete']
    os.kill(r['pid'], signal.SIGSTOP)  # CPU parent only; never its process group or GPU child.
    time.sleep(.15)
    now = same_process(state)
    assert now and now['state'] in ['T','t']
    state.update(phase='handoff_child_running_parent_paused', pause_time=time.time(),
                 parent_at_pause=identity(r), handoff_child=identity(match[0]))
    write(S/f'audits/balanced_ood_gpu{gpu}_parent_pause.json', dict(time=time.time(),
        execution_amendment_sha256=ah, gpu=gpu, state=state,
        scope='Only CPU dispatch parent SIGSTOP. Existing frozen evaluation child continues unchanged; GPU3 full MATH, GPU2 baseline/nonzero dev replay.'))

def retire_parent(gpu):
    state = parents[gpu]; r = same_process(state)
    assert r and r['state'] in ['T','t']
    os.kill(r['pid'], signal.SIGTERM)
    os.kill(r['pid'], signal.SIGCONT)
    for _ in range(50):
        current = same_process(state)
        if not current or current['state']=='Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('CPU dispatch parent did not retire')
    state.update(phase='available', needs_manual_dispatch_recovery=False, retired_time=time.time(), retirement='SIGTERM then SIGCONT, after handoff child exit0 and two quiet GPU checks')
    write(S/f'audits/balanced_ood_gpu{gpu}_handoff.json', dict(time=time.time(),
        execution_amendment_sha256=ah, state=state, passed=True))

def verify_gpu2_replay():
    actual = S/'audits/expanded_supplement_gpu2_protocol_replay.json'
    legacy = S/'audits/expanded_supplement_gpu1_protocol_replay.json'
    assert actual.exists() and legacy.exists()
    previous=json.loads(actual.read_text())
    prior_legacy=json.loads(legacy.read_text())
    assert previous['passed'] and previous['actual_gpu']=='2' and previous['n']==1000
    assert previous['plan_sha256']==amend['supplement_plan_sha256']
    assert prior_legacy['passed'] and prior_legacy['actual_gpu']=='2'
    assert prior_legacy['actual_replay_sha256']==sha(actual)
    original = json.loads((S/'results/confirmation_manifest.json').read_text())
    fixed43 = plan['families'][plan['primary_family']]['43']
    expected = {'supplement_gpu2_replay_base':None, 'supplement_gpu2_replay_nonzero':original[fixed43]}
    replay_manifest = S/'results/expanded_supplement_replay_manifest.json'
    assert json.loads(replay_manifest.read_text()) == expected
    matches=[]; fields=['index','sample_hash','prediction','correct','finish_reason','generated_tokens']
    for alias, reference in [('supplement_gpu2_replay_base','base'),('supplement_gpu2_replay_nonzero',fixed43)]:
        root=S/'results/evaluation/dev'; x=root/alias/'math_predictions.jsonl'; y=root/reference/'math_predictions.jsonl'
        assert (root/alias/'math_summary.json').exists()
        a=read(x);b=read(y);assert len(a)==len(b)==500
        assert all(all(u[k]==v[k] for k in fields) for u,v in zip(a,b)), (alias,reference)
        matches.append(dict(alias=alias,reference=reference,n=500,alias_sha256=sha(x),reference_sha256=sha(y),fields=fields))
    assert previous['matches']==matches
    assert prior_legacy['matches']==matches
    write(S/'audits/balanced_gpu2_existing_replay_reuse.json',dict(time=time.time(),passed=True,
        execution_amendment_sha256=ah,actual_replay_sha256=sha(actual),legacy_record_sha256=sha(legacy),
        scope='Independently rechecked all1000 frozen six-field comparisons; reused actual earlier GPU2 replay without regenerating or overwriting its audit.'))


def advance_parent(gpu):
    state = parents[gpu]
    if state['phase']=='waiting_handoff_child':
        capture_math_parent(gpu)
    if state['phase']=='handoff_child_running_parent_paused':
        child = same_process(state['handoff_child'])
        if not child:
            raise RuntimeError((gpu, 'handoff child lost before verified exit'))
        assert child['ppid'] == state['pid'], (gpu, 'handoff parent identity changed')
        if child['state'] != 'Z':
            return
        assert child['exit_code']==0, (gpu, 'handoff child exit', child['exit_code'])
        if gpu=='3':
            assert summaries_complete('math_reused5000', main_names)
        else:
            verify_gpu2_replay()
            state['protocol_replay_sha256']=sha(S/'audits/expanded_supplement_gpu2_protocol_replay.json')
            assert summaries_complete('math_reused5000',supp_names)
        state.update(phase='draining_after_handoff_child', child_exit=identity(child), child_exit_time=time.time())
    if state['phase']=='draining_after_handoff_child':
        check = gpu_quiet(gpu)
        state['last_drain_check'] = check
        state['quiet'] = state['quiet']+1 if check['quiet'] else 0
        if state['quiet'] >= 2:
            retire_parent(gpu)
            if gpu=='2':
                frozen=json.loads((S/'audits/expanded_unaveraged_supplement_frozen.json').read_text())
                for task in tasks:
                    if task['role']!='supplement' or task['dataset']!='math_reused5000':
                        continue
                    assert task['status']=='pending' and len(task['names'])==1
                    name=task['names'][0];root=S/'results/evaluation/math_reused5000'/name
                    m=json.loads((root/'math_manifest.json').read_text())
                    for key,value in plan['evaluation_protocol'].items():assert m[key]==value
                    assert m['data_sha256']==plan['data_sha256']['math_reused5000']
                    assert m['adapter']['files']==frozen['artifacts'][name]['files']
                    task.update(status='complete',gpu='2',reused_original_math=True,
                        observed_original_child=state['handoff_child'],
                        summary_sha256=sha(root/'math_summary.json'),manifest_sha256=sha(root/'math_manifest.json'),
                        completion_source='Original GPU2 five-model MATH invocation finished naturally with exit0; no extra per-model invocation launched.')

def start_task(gpu, task):
    assert time.time()<finish and sha(ap)==ah
    assert sha(S/'scripts/evaluate.py') == amend['unchanged_scripts']['evaluate.py']
    manifest = S/task['manifest']
    assert sha(manifest)==task['manifest_sha256']
    command = [sys.executable]
    if gpu=='2':
        assert json.loads((S/'audits/expanded_supplement_gpu2_protocol_replay.json').read_text())['passed']
        command.append(str(S/'scripts/gpu2_authorized_entry.py'))
    command += [str(S/'scripts/evaluate.py'),'--dataset',task['dataset'],'--manifest',str(manifest)]
    handle = (S/f'logs/balanced_ood_{task["id"]}.log').open('w')
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), start_new_session=True)
    running[gpu] = (process, task); logs[gpu] = handle
    task.update(status='running', gpu=gpu, pid=process.pid, start=time.time(), command=command)
    parents[gpu]['quiet']=0
    with (S/'logs/commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(),gpu=gpu,command=command,role='balanced_frozen_ood',execution_amendment_sha256=ah))+'\n')

def finish_task(gpu):
    process, task = running[gpu]; rc = process.poll()
    if rc is None:
        return
    logs.pop(gpu).close(); del running[gpu]
    okay = rc==0 and summaries_complete(task['dataset'],task['names'])
    task.update(status='complete' if okay else 'error', returncode=rc, end=time.time())
    calls.append(dict(task_id=task['id'],gpu=gpu,returncode=rc,start=task['start'],end=task['end']))
    if not okay:
        errors.append(dict(time=time.time(),task_id=task['id'],returncode=rc))

def stop_new_children():
    for gpu,(process,task) in list(running.items()):
        if process.poll() is None:
            assert os.getpgid(process.pid)==process.pid
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)
        finish_task(gpu)

publish('waiting_handoffs')
try:
    while time.time()<finish:
        assert sha(ap)==ah
        for gpu in ['3','2']:
            advance_parent(gpu)
            if gpu in running:
                finish_task(gpu)
            if parents[gpu]['phase']!='available' or gpu in running:
                continue
            check=gpu_quiet(gpu); parents[gpu]['last_drain_check']=check
            parents[gpu]['quiet']=parents[gpu]['quiet']+1 if check['quiet'] else 0
            if parents[gpu]['quiet']<2:
                continue
            task=next((t for t in tasks if t['status']=='pending'),None)
            if task:
                start_task(gpu,task)
        if all(t['status']=='complete' for t in tasks) and not running and all(s['phase']=='available' and s['quiet']>=2 for s in parents.values()):
            publish('all_gpu_evaluations_complete');break
        if errors and all(t['status'] in ['complete','error'] for t in tasks) and not running and all(s['phase']=='available' for s in parents.values()):
            publish('incomplete_error_or_cutoff');break
        publish('running_or_waiting_handoff');time.sleep(5)
    else:
        stop_new_children()
        # Never resume an obsolete dispatcher automatically into possibly partial OOD outputs.
        # Root must reconcile any unfinished MATH child; the independent global deadline guard remains active.
        for state in parents.values():
            r=same_process(state)
            if r and r['state'] in ['T','t']:
                state['needs_manual_dispatch_recovery']=True
        publish('incomplete_error_or_cutoff')
except BaseException as exc:
    errors.append(dict(time=time.time(),error=repr(exc)))
    stop_new_children()
    for state in parents.values():
        r=same_process(state)
        if r and r['state'] in ['T','t']:
            state['needs_manual_dispatch_recovery']=True
    publish('error')
    raise
