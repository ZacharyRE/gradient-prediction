"""Execute already fixed native controls and supplementary models on authorized GPU2."""
import subprocess
import signal
from common import *

ap=S/'audits/gpu2_additional_execution_amendment.json';amend=json.loads(ap.read_text());ah=sha(ap)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
assert amend['maximum_simultaneous_gpus']==3 and amend['execution_script_sha256']==sha(__file__)
assert amend['entry_script_sha256']==sha(S/'scripts/gpu2_authorized_entry.py')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
np=S/'audits/native_target_control_plan.json';native=json.loads(np.read_text());nh=sha(np)
sp=S/'audits/expanded_unaveraged_supplement_plan.json';supp=json.loads(sp.read_text());sh=sha(sp)
fp=S/'audits/expanded_unaveraged_supplement_frozen.json';frozen=json.loads(fp.read_text())
primary_path=S/'results/confirmation_plan.json';primary=json.loads(primary_path.read_text())
assert nh==amend['native_plan_sha256'] and sh==amend['supplement_plan_sha256']
assert sha(fp)==amend['supplement_frozen_sha256'] and frozen['primary_plan_sha256']==sha(primary_path)==amend['primary_plan_sha256']
assert sha(S/'scripts/analyze_expanded_supplement.py')==frozen['analysis_sha256']
for filename,digest in amend['unchanged_inference_scripts'].items():assert sha(S/'scripts'/filename)==digest
assert sha(S/'scripts/common.py')==amend['unchanged_common_sha256']
assert not (S/'audits/native_target_control_status.json').exists()
assert not (S/'audits/expanded_unaveraged_supplement_status.json').exists()
finish=amend['unchanged_finish_cutoff'];calls=[]
def audit(path,record):
    write(path,dict(record,execution_amendment_sha256=ah,execution_script_sha256=sha(__file__),actual_gpu='2'))
def guard():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2' and sha(ap)==ah
    assert time.time()<min(DEADLINE-60,finish) and not (S/'STOP_GPU2_ADDITIONAL').exists()
def drain(label):
    stable=0;checks=[]
    while time.time()<min(DEADLINE-60,finish):
        own=[]
        for line in subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
            if not line.strip().isdigit():continue
            try:
                if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():own.append(int(line))
            except FileNotFoundError:pass
        free,total=map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
        good=not own and free>=max(45*1024,.30*total+2048)
        stable=stable+1 if good else 0;checks.append(dict(time=time.time(),own_gpu_pids=own,free_mib=free,quiet=good))
        if stable>=2:
            audit(S/f'audits/gpu2_drain_{label}.json',dict(checks=checks,passed=True));return
        time.sleep(5)
    raise RuntimeError('GPU2 natural drain cutoff')
def run(script,args,label):
    guard();assert sha(S/'scripts'/script)==amend['unchanged_inference_scripts'][script]
    command=[sys.executable,str(S/'scripts/gpu2_authorized_entry.py'),str(S/'scripts'/script),*args]
    with (S/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu='2',command=command,role='authorized_additional'))+'\n')
    with (S/f'logs/gpu2_{label}.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=proc.wait(timeout=max(1,min(DEADLINE-60,finish)-time.time()))
        except subprocess.TimeoutExpired:
            assert os.getpgid(proc.pid)==proc.pid
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            rc=124
    calls.append(dict(time=time.time(),label=label,command=command,returncode=rc))
    audit(S/'audits/gpu2_additional_dispatch.json',dict(calls=calls,all_complete=False))
    assert rc==0,(label,rc)
    drain(label)
def replay(alias,reference,backend,n):
    filename='predictions.jsonl' if backend=='native' else 'math_predictions.jsonl'
    root=S/('results/native/dev' if backend=='native' else 'results/evaluation/dev')
    x=root/alias/filename;y=root/reference/filename;aa=read(x);bb=read(y)
    assert len(aa)==len(bb)==n
    fields=['index','sample_hash','prediction','correct','finish_reason','generated_tokens']
    assert all(all(a[k]==b[k] for k in fields) for a,b in zip(aa,bb)),(alias,reference)
    return dict(alias=alias,reference=reference,n=n,alias_sha256=sha(x),reference_sha256=sha(y),fields=fields)

stage='native'
try:
    guard();drain('initial')
    audit(S/'audits/gpu2_additional_handoff.json',dict(time=time.time(),passed=True,scope='User raised GPU limit to3. Only two CPU waiters were retired; original GPU1 native and GPU3 benchmarks continue unchanged.'))
    # Verify both reused native operands on the new physical GPU before new controls.
    oldgen='sample_all_lr1e5_s43_epoch2'
    oldmanifest=json.loads((S/'results/native/dev'/oldgen/'manifest.json').read_text())
    genpath=S/'results/training/sample_all_lr1e5_s43/epoch2'
    assert sha(genpath/'adapter_model.safetensors')==oldmanifest['adapter']['files']['adapter_model.safetensors']
    replay_manifest={'gpu2_native_replay_base':None,'gpu2_native_replay_student':str(genpath)}
    rp=S/'results/gpu2_native_replay_manifest.json';write(rp,replay_manifest)
    audit(S/'audits/native_target_control_status.json',dict(time=time.time(),status='running_gpu2_protocol_replay',all_complete=False,plan_sha256=nh))
    run('native_recheck.py',['--manifest',str(rp),'--limit','128','--batch','8'],'native_protocol_replay')
    matches=[replay('gpu2_native_replay_base','base','native',128),replay('gpu2_native_replay_student',oldgen,'native',128)]
    audit(S/'audits/gpu2_native_protocol_replay.json',dict(time=time.time(),passed=True,n=256,matches=matches,native_plan_sha256=nh))
    manifest={}
    for name,e in native['artifacts'].items():
        p=Path(e['path']);assert sha(p/'adapter_model.safetensors')==e['weight_sha256'] and sha(p/'adapter_config.json')==e['config_sha256']
        assert not (S/'results/native/dev'/name).exists();manifest[name]=str(p)
    mp=S/'results/native_target_control_manifest.json';write(mp,manifest)
    audit(S/'audits/native_target_control_handoff.json',dict(time=time.time(),plan_sha256=nh,manifest_sha256=sha(mp),protocol_replay_sha256=sha(S/'audits/gpu2_native_protocol_replay.json')))
    audit(S/'audits/native_target_control_status.json',dict(time=time.time(),status='running',all_complete=False,plan_sha256=nh))
    run('native_recheck.py',['--manifest',str(mp),'--limit','128','--batch','8'],'native_target_controls')
    assert all((S/'results/native/dev'/n/'summary.json').exists() for n in native['new_models'])
    audit(S/'audits/native_target_control_status.json',dict(time=time.time(),status='all_gpu_evaluations_complete',all_complete=True,plan_sha256=nh,calls=list(calls)))
    stage='supplement'
    sm=S/'results/expanded_unaveraged_supplement_manifest.json';assert sha(sm)==frozen['manifest_sha256']
    for name,e in frozen['artifacts'].items():
        for filename,digest in e['files'].items():assert sha(Path(e['adapter'])/filename)==digest
    for dataset in supp['datasets']:assert sha(S/f'data/{dataset}.jsonl')==supp['data_sha256'][dataset]
    primary43=primary['families'][primary['primary_family']]['43'];allmodels=json.loads((S/'results/confirmation_manifest.json').read_text())
    rp=S/'results/expanded_supplement_replay_manifest.json'
    write(rp,{'supplement_gpu2_replay_base':None,'supplement_gpu2_replay_nonzero':allmodels[primary43]})
    audit(S/'audits/expanded_unaveraged_supplement_handoff.json',dict(time=time.time(),plan_sha256=sh,frozen_sha256=sha(fp),scope='Same five frozen models, advanced on GPU2 after native controls; primaryGPU1/GPU3 work continues.'))
    audit(S/'audits/expanded_unaveraged_supplement_status.json',dict(time=time.time(),status='running_gpu2_protocol_replay',all_complete=False,plan_sha256=sh))
    run('evaluate.py',['--dataset','dev','--manifest',str(rp)],'supplement_protocol_replay')
    matches=[replay('supplement_gpu2_replay_base','base','vllm',500),replay('supplement_gpu2_replay_nonzero',primary43,'vllm',500)]
    actual=S/'audits/expanded_supplement_gpu2_protocol_replay.json'
    audit(actual,dict(time=time.time(),passed=True,n=1000,matches=matches,plan_sha256=sh))
    # Frozen analyzer uses this historical filename; actual physicalGPU and source are explicit.
    audit(S/'audits/expanded_supplement_gpu1_protocol_replay.json',dict(time=time.time(),passed=True,n=1000,plan_sha256=sh,
        legacy_filename_only=True,original_planned_gpu='1',actual_replay_path=str(actual),actual_replay_sha256=sha(actual),matches=matches))
    for dataset in supp['datasets']:
        run('evaluate.py',['--dataset',dataset,'--manifest',str(sm)],'supplement_'+dataset)
        audit(S/'audits/expanded_unaveraged_supplement_status.json',dict(time=time.time(),status='running',all_complete=False,plan_sha256=sh,calls=list(calls)))
    audit(S/'audits/expanded_unaveraged_supplement_status.json',dict(time=time.time(),status='all_gpu_evaluations_complete',all_complete=True,plan_sha256=sh,calls=list(calls)))
    audit(S/'audits/gpu2_additional_dispatch.json',dict(time=time.time(),calls=calls,all_complete=True))
except BaseException as exc:
    if stage=='native':
        audit(S/'audits/native_target_control_status.json',dict(time=time.time(),status='incomplete_error_or_cutoff',all_complete=False,plan_sha256=nh,error=repr(exc)))
        audit(S/'audits/expanded_unaveraged_supplement_status.json',dict(time=time.time(),status='not_started_native_gpu2_error',all_complete=False,plan_sha256=sh))
    else:
        audit(S/'audits/expanded_unaveraged_supplement_status.json',dict(time=time.time(),status='incomplete_error_or_cutoff',all_complete=False,plan_sha256=sh,error=repr(exc)))
    raise
