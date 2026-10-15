from pathlib import Path
import hashlib, json, os, random, re, time, shutil, subprocess
import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
os.environ['PATH'] = str(ROOT/'.venv/bin') + os.pathsep + os.environ.get('PATH','')
MODELS = {'1p5b':os.environ.get('PREDICTOR_MODEL_PATH','/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')}
SYSTEM = 'You are a helpful assistant.'
USER = 'Solve the following mathematics problem. Show your reasoning clearly and place only the final answer inside \\boxed{{}}.\n\nProblem:\n{problem}'
TASKS = ['mnli','paws','hellaswag','race']

def read(p):
    with Path(p).open() as f: return [json.loads(x) for x in f if x.strip()]
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def jsonl(p, rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def source_signature(entrypoint):
    sources={p.name:sha(p) for p in sorted((EXP/'scripts').glob('*.py'))}
    return hashlib.sha256(json.dumps(dict(entrypoint=Path(entrypoint).name,sources=sources),sort_keys=True).encode()).hexdigest()
def identity(r):return hashlib.sha256(re.sub(r'\s+','',r['problem']).encode()).hexdigest()
def data(task,split):return EXP/'data'/task/f'{split}.jsonl'
def log(d):print(json.dumps(d,ensure_ascii=False),flush=True)
def deadline(reserve=0):
    # Frozen timing/profiling replays are validation, not new recipe searches.
    # They still stop before the absolute GPU cutoff, with ten minutes to clean up.
    if os.environ.get('BENCH_FINAL_VALIDATION')=='1':reserve=min(reserve,600)
    return time.time()+reserve>=json.loads((EXP/'configs/run_deadline.json').read_text())['deadline_epoch']
def guard_space(min_gib=20):
    # Bulk outputs may live on a separate volume; control files remain local.
    layout=EXP/'configs/storage_layout.json'
    policy=json.loads(layout.read_text()) if layout.exists() else None
    if policy:
        assert policy['mode']=='bulk_outputs_on_owned_shared_volume'
        for relative,destination in policy['bulk_roots'].items():
            assert (EXP/relative).is_symlink() and str((EXP/relative).resolve())==destination
        control_roots=policy.get('control_roots')
        if control_roots:
            for relative,destination in control_roots.items():
                assert (EXP/relative).is_symlink() and str((EXP/relative).resolve())==destination
            locations=[(EXP/p,policy['control_minimum_free_gib']) for p in control_roots]
        else:locations=[(EXP,policy['control_minimum_free_gib'])]
        locations += [(EXP/p,min_gib) for p in policy['bulk_roots']]
    else:locations=[(EXP,min_gib),(EXP/'cache',min_gib)]
    devices={};available=[]
    for location,minimum in locations:
        if not location.exists():continue
        location=location.resolve();device=location.stat().st_dev
        devices[device]=(location,max(minimum,devices.get(device,(None,0))[1]))
    for location,minimum in devices.values():
        free=shutil.disk_usage(location).free/1024**3
        if free<minimum:raise RuntimeError(f'Disk guard at {location}: {free:.1f} GiB; need {minimum}. Never delete old experiment data.')
        available.append(free)
    return min(available)
def incidental_context_allowed(pid,memory_mib):
    path=EXP/'configs/incidental_context_policy.json'
    if not path.exists():return False
    policy=json.loads(path.read_text())
    for item in policy['allowed']:
        if pid==item['pid'] and process_start(pid)==item['process_start']:
            return float(memory_mib)<=item['max_memory_mib']
    return False

def gpu_guard():
    active=EXP/'results/active_gpu.json'
    cfg=json.loads((active if active.exists() else EXP/'configs/protocol.json').read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['gpu_index']):raise RuntimeError('Unexpected physical GPU assignment')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    holder=reservation_state()
    permitted=holder['pid'] if holder and holder['gpu_uuid']==cfg['gpu_uuid'] and holder['mode']=='stage' else None
    if cfg.get('allocation_mode')=='shared':
        import torch
        policy=json.loads((EXP/'configs/shared_gpu_policy.json').read_text())
        torch.cuda.set_per_process_memory_fraction(policy['shared_max_worker_fraction'],0)
        return
    foreign=[line for line in output.splitlines() if line.split(',')[0].strip()==cfg['gpu_uuid'] and int(line.split(',')[1])!=permitted and not incidental_context_allowed(int(line.split(',')[1]),line.split(',')[2])]
    if foreign:raise RuntimeError('GPU occupied at stage boundary: '+str(foreign))
def process_start(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,PermissionError,ProcessLookupError):return None
def reservation_state():
    path=EXP/'results/gpu_reservation.json'
    if not path.exists():return None
    state=json.loads(path.read_text());pid=state.get('pid');proc=Path('/proc')/str(pid)
    if state.get('status')!='ready' or process_start(pid)!=state.get('process_start'):return None
    try:
        env=(proc/'environ').read_bytes().split(b'\0') if proc.stat().st_uid==os.getuid() else []
    except (FileNotFoundError,PermissionError,ProcessLookupError):return None
    return state if ('BENCH_RUN_ID='+EXP.name).encode() in env and b'BENCH_GPU_RESERVER=1' in env else None
def mean_gain_mode():
    path=EXP/'configs/oracle_admission_policy.json'
    return path.exists() and json.loads(path.read_text()).get('mode')=='three_seed_mean_gain'

def refresh_mean_admission(screen):
    """Reclassify admission; preserve original criteria and scientific observations."""
    if not mean_gain_mode():return screen
    if 'admission_before_user_mean_rule' not in screen:
        screen['admission_before_user_mean_rule']={k:screen.get(k) for k in ['gate_passed','development_gate_passed','gate_basis']}
    acc=screen.get('replicate_dev_accuracies',[])
    complete=len(acc)==3 and len(screen.get('selected_oracles',[]))==3
    gain=100*(float(np.mean(acc))-screen['warmup']['dev_accuracy']) if complete else None
    dev=bool(complete and gain>0)
    if dev and screen.get('additional_warmup_controls'):
        dev=bool(float(np.mean(acc))>screen['primary_warmup_dev_accuracy'])
    screen.update(development_gate_passed=dev,mean_development_gain_pp=gain,admission_policy_revision='user_mean_gain_v1')
    confirmation=screen.get('confirmation',{})
    if confirmation:
        confirmation.setdefault('passed_before_user_mean_rule',confirmation.get('passed'))
        stats=confirmation['oracle_vs_warmup']
        confirmation['passed']=bool(stats['seeds']==3 and stats['mean_pp']>0)
        if screen.get('additional_warmup_controls'):
            extra=screen.get('oracle_vs_primary_warmup_confirmation',{})
            confirmation['passed']=bool(confirmation['passed'] and extra.get('seeds')==3 and extra.get('mean_pp',0)>0)
        confirmation['criterion']='Three update-seed mean accuracy greater than the same fixed warmup; no per-seed, >=2pp, base-model or confidence-interval admission threshold'
        confirmation['admission_policy_revision']='user_mean_gain_v1'
    screen['gate_passed']=bool(dev and confirmation.get('passed'))
    screen['gate_basis']='User mean-gain rule: three-seed average above fixed warmup on development and independent confirmation; uncertainty and individual seeds remain reported'
    return screen

def require_confirmed_oracle(task):
    screen=json.loads((EXP/'results'/f'{task}_oracle_screen.json').read_text())
    confirmation=screen.get('confirmation',{})
    if mean_gain_mode():
        assert screen.get('admission_policy_revision')=='user_mean_gain_v1'
        assert screen['gate_passed'] and screen['development_gate_passed'] and confirmation.get('passed')
        assert len(screen['selected_oracles'])==3 and len(screen['replicate_dev_accuracies'])==3 and confirmation['n']>=32
        assert float(np.mean(screen['replicate_dev_accuracies']))>screen['warmup']['dev_accuracy']
        assert confirmation['oracle_vs_warmup']['seeds']==3 and confirmation['oracle_vs_warmup']['mean_pp']>0
        if screen.get('additional_warmup_controls'):
            assert screen['oracle_vs_primary_warmup_confirmation']['mean_pp']>0
    else:
        interval=confirmation.get('interval','ci95_pp')
        if screen.get('followup_confirmation',{}).get('passed'):
            confirmation=screen['followup_confirmation'];interval='ci97p5_two_look_pp'
        assert screen.get('gate_passed') and confirmation.get('passed'),f'{task}: independent oracle confirmation must pass before predictor work'
        assert len(screen['selected_oracles'])==3 and confirmation['n']>=32
        for contrast in ['oracle_vs_warmup','oracle_vs_base']:
            stats=confirmation[contrast]
            assert stats['seeds']==3 and stats['mean_pp']>=2 and stats[interval][0]>0,(task,contrast,stats)
        if screen.get('additional_warmup_controls'):
            stats=screen['oracle_vs_primary_warmup_confirmation']
            assert stats['seeds']==3 and stats['mean_pp']>=2 and stats[interval][0]>0,(task,'stronger primary warmup',stats)
    warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    oracle=json.loads((EXP/'configs'/f'{task}_oracle.json').read_text())
    assert warm['adapter_sha256']==screen['warmup']['adapter_sha256']
    assert oracle['adapter_sha256']==screen['oracle']['adapter_sha256']
    return confirmation
def reserve_worker_cache(_worker=None,free_gib=3):
    """Keep unused VRAM in this worker's reusable PyTorch allocator cache.

    Empty tensors are released to the allocator, not kept as live tensors, so
    subsequent model allocations can reuse the same memory. No kernels, RNG,
    model weights, prompts or evaluation settings are changed.
    """
    import torch
    assert os.environ.get('BENCH_RUN_ID')==EXP.name
    assert torch.cuda.device_count()==1
    assignment=json.loads((EXP/'results/active_gpu.json').read_text())
    if assignment.get('allocation_mode')=='shared':
        policy=json.loads((EXP/'configs/shared_gpu_policy.json').read_text())
        torch.cuda.set_per_process_memory_fraction(policy['shared_max_worker_fraction'],0)
        free,total=torch.cuda.mem_get_info()
        memory=dict(free_bytes=free,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
        result=dict(time=time.time(),pid=os.getpid(),process_start=process_start(os.getpid()),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),before=memory,after=memory,seconds=0.0,padding_allocation_oom=False,role='Shared GPU: no padding allocation; per-process allocator fraction cap',fraction_cap=policy['shared_max_worker_fraction'])
        write(EXP/'results/worker_memory_reservations'/f'{os.getpid()}.json',result)
        return result
    # Larger reusable blocks also accommodate the ~3GiB finite gradient table
    # without forcing the allocator to flush its entire unused reservation.
    gib=1024**3;keep=int(free_gib*gib);chunk=8*gib;blocks=[]
    free,total=torch.cuda.mem_get_info();before=dict(free_bytes=free,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    start=time.perf_counter();oom=False
    try:
        # Existing cache blocks may be reused first; remeasure physical free
        # memory on each iteration rather than assuming every block is new.
        for _ in range(int(total/gib)+1):
            free,_=torch.cuda.mem_get_info();size=min(chunk,free-keep)
            if size<64*1024**2:break
            try:blocks.append(torch.empty(size,dtype=torch.uint8,device='cuda'))
            except torch.OutOfMemoryError:oom=True;break
    finally:blocks.clear()
    free,_=torch.cuda.mem_get_info()
    result=dict(time=time.time(),pid=os.getpid(),process_start=process_start(os.getpid()),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),free_target_gib=free_gib,before=before,after=dict(free_bytes=free,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()),seconds=time.perf_counter()-start,padding_allocation_oom=oom,role='reusable worker cache, not model memory or live tensors')
    write(EXP/'results/worker_memory_reservations'/f'{os.getpid()}.json',result)
    return result
def exploration_arms():
    path=EXP/'configs/oracle_setup_exploration.json'
    return json.loads(path.read_text())['arms'] if path.exists() else {}

def task_config(task):
    cfg=json.loads((EXP/'configs/protocol.json').read_text())
    cfg.update(exploration_arms().get(task,{}).get('training_overrides',{}))
    return cfg

def output_cap(task):
    task=exploration_arms().get(task,{}).get('source_task',task)
    return 2048 if task in ['gsm8k','math','svamp','aqua','countdown'] else 256

def prompt(tok,row):
    task=row['task'];kind=row.get('answer_type','math');system=SYSTEM
    if row.get('prompt_variant')=='aqua_neutral':
        system='You are a helpful assistant and an expert mathematician.'
        user='Solve the following multiple-choice mathematics problem. Put the final option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='mnli':user='Determine the relationship between the premise and hypothesis. A means entailment, B means neutral, and C means contradiction. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='paws':user='Determine whether the two sentences have the same meaning. Return only yes or no inside \\boxed{}.\n\n'+row['problem']
    elif task=='hellaswag':user='Choose the most plausible continuation of the context. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='commonsense_qa':user='Answer the following multiple-choice commonsense question. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='social_i_qa':user='Answer the multiple-choice question about the social situation. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='winogrande':user='Choose the option that correctly fills the blank in the sentence. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='sst2':user='Classify the sentiment of the sentence. A means negative and B means positive. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='sciq':user='Answer the following multiple-choice science question. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='race':user='Answer the multiple-choice question using the passage. Return only the correct option letter inside \\boxed{}.\n\n'+row['problem']
    elif task=='aqua':
        system='You are a helpful assistant and an expert mathematician.'
        user='Solve the following multiple-choice mathematics problem. Show your reasoning and put the final option letter inside \\boxed{}.\n\n'+row['problem']
    elif kind=='choice':user='Answer the following multiple-choice science question. Put the final option letter inside \\boxed{}.\n\n'+row['problem']
    elif kind=='bool':user='Answer the question using the passage. Return only yes or no inside \\boxed{}.\n\n'+row['problem']
    else:
        system='You are a helpful assistant and an expert mathematician.'
        user=USER.format(problem=row['problem'])
    return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
