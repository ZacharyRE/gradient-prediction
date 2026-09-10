"""Three bounded native target controls, after both existing GPU1 owners finish."""
import signal
import subprocess
from common import *
gpu_guard(); assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
pp=S/'audits/native_target_control_plan.json';plan=json.loads(pp.read_text());ph=sha(pp)
out=S/'audits/native_target_control_status.json'
def predecessors_done():
    native=S/'audits/final_native_dispatch.json';supp=S/'audits/expanded_unaveraged_supplement_status.json'
    if not native.exists() or not supp.exists():return False
    n=json.loads(native.read_text());v=json.loads(supp.read_text())
    return n.get('all_complete') and (v.get('all_complete') or v.get('status','').startswith('not_started') or v.get('status')=='incomplete_error_or_cutoff')
while not predecessors_done():
    if time.time()>=plan['start_cutoff']:
        write(out,dict(time=time.time(),status='not_started_predecessor_cutoff',all_complete=False,plan_sha256=ph));raise SystemExit(0)
    time.sleep(10)
stable=0;checks=[]
while time.time()<plan['start_cutoff']:
    owned=[]
    for line in subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
        if not line.strip().isdigit():continue
        try:
            if (Path('/proc')/line.strip()).stat().st_uid==os.getuid():owned.append(int(line))
        except FileNotFoundError:pass
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    good=not owned and free>=45*1024;stable=stable+1 if good else 0
    checks.append(dict(time=time.time(),own_gpu_pids=owned,free_mib=free,quiet=good))
    if stable>=2:break
    time.sleep(5)
else:
    write(out,dict(time=time.time(),status='not_started_gpu_drain_cutoff',all_complete=False,plan_sha256=ph,checks=checks));raise SystemExit(0)
assert (S/'STOP_TRAINER').exists() and sha(pp)==ph
assert sha(S/'scripts/native_recheck.py')==plan['native_script_sha256']
assert sha(S/'data/dev.jsonl')==plan['data_sha256']
manifest={}
for name,evidence in plan['artifacts'].items():
    path=Path(evidence['path']);assert sha(path/'adapter_model.safetensors')==evidence['weight_sha256']
    assert sha(path/'adapter_config.json')==evidence['config_sha256']
    assert not (S/'results/native/dev'/name/'summary.json').exists()
    manifest[name]=str(path)
mp=S/'results/native_target_control_manifest.json';write(mp,manifest)
write(S/'audits/native_target_control_handoff.json',dict(time=time.time(),plan_sha256=ph,
    checks=checks,manifest_sha256=sha(mp),pipeline_sha256=sha(__file__),
    scope='Both original native and supplementaryGPU1 work finished naturally; no existing process signaled.'))
command=[sys.executable,str(S/'scripts/native_recheck.py'),'--manifest',str(mp),'--limit','128','--batch','8']
with (S/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu='1',command=command,role='native_target_control'))+'\n')
with (S/'logs/native_target_control_inference.log').open('w') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out,dict(time=time.time(),status='running',all_complete=False,plan_sha256=ph,pid=process.pid,command=command))
    try:rc=process.wait(timeout=max(1,min(DEADLINE-60,plan['finish_cutoff'])-time.time()))
    except subprocess.TimeoutExpired:
        assert os.getpgid(process.pid)==process.pid
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        rc=124
complete=rc==0 and all((S/'results/native/dev'/n/'summary.json').exists() for n in plan['new_models'])
write(out,dict(time=time.time(),status='all_gpu_evaluations_complete' if complete else 'incomplete_error_or_cutoff',
    all_complete=complete,plan_sha256=ph,command=command,returncode=rc,
    scope='Native inference completion only; paired scoring and review still required.'))
assert complete,(rc,complete)
