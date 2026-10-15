"""Prepare and run one selected recipe in a fresh directory, with optional dev evaluation.

Requires the recorded source-study data/warmup artifacts and a local Qwen model.
Original neural predictors also require their recorded checkpoint; --predictor-
checkpoint overrides a recipe's path. No GPU work occurs with --prepare-only.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,hashlib,signal

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-study',type=Path,required=True)
    ap.add_argument('--model-path',type=Path,required=True)
    ap.add_argument('--jobs',type=Path,required=True)
    ap.add_argument('--recipe',required=True)
    ap.add_argument('--module',required=True,choices=['research','research_v3','research_v4','research_fastio','research_h','research_fastio_h','research_state','research_multistate','research_lambda','research_teacher_cache','research_cached_head_fastio','anchored','anchored_fastio','stratified','residual_ema','online_ridge','online_ridge_v2','memory_cv','time_query'])
    ap.add_argument('--task',choices=['paws','mnli'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--seed',type=int,default=101)
    ap.add_argument('--predictor-seed',type=int,default=123)
    ap.add_argument('--predictor-checkpoint',type=Path)
    ap.add_argument('--warmup-adapter',type=Path,help='Relocated copy of the recorded warmup adapter; its weights must match the recorded SHA256')
    ap.add_argument('--steps',type=int)
    ap.add_argument('--duration-minutes',type=int,default=60)
    ap.add_argument('--compute-cutoff-epoch',type=float,help='Optional absolute Unix-time cap for GPU work; can only shorten the duration budget')
    ap.add_argument('--evaluate',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();assert a.duration_minutes>=15 and a.gpu>=0
    scripts=Path(__file__).resolve().parent;module_file=scripts/(a.module+'.py')
    assert module_file.is_file() and module_file.parent==scripts,'Choose a provided trainer module'
    source=a.source_study.resolve();model=a.model_path.resolve();output=a.output.resolve()
    assert source.is_dir() and (model/'config.json').exists()
    assert not output.exists(),'Use a new output directory; existing experiments are preserved'
    matches=[j for j in json.loads(a.jobs.read_text()) if j['name']==a.recipe]
    assert len(matches)==1;job=dict(matches[0])
    if a.steps is not None:
        assert a.steps>0;job['steps']=a.steps
    job['save']=sorted({s for s in job.get('save',[]) if s<=job['steps']}|{job['steps']})
    if a.predictor_checkpoint:
        assert a.predictor_checkpoint.is_file();job['predictor_path']=str(a.predictor_checkpoint.resolve())
    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    warm=json.loads((source/'configs'/f'{a.task}_warmup.json').read_text())
    if a.warmup_adapter:
        warm['adapter']=str(a.warmup_adapter.resolve())
    adapter=Path(warm['adapter'])/'adapter_model.safetensors'
    assert adapter.exists() and digest(adapter)==warm['adapter_sha256'],'Recorded warmup weights must match'
    for name in ['scripts','configs','results','models','cache','logs','data','provenance']:(output/name).mkdir(parents=True,exist_ok=True)
    for path in scripts.glob('*.py'):shutil.copy2(path,output/'scripts'/path.name)
    (output/'data'/a.task).symlink_to(source/'data'/a.task,target_is_directory=True)
    artifact_source=source
    inherited=source/'configs/protocol.json'
    suffix='_globalfit' if a.task=='mnli' else ''
    inherited_name=f'{a.task}_predictor{suffix}_p{a.predictor_seed}'
    if not (source/'models'/inherited_name/'best.pt').exists() and inherited.exists():
        recorded=json.loads(inherited.read_text()).get('source_study')
        if recorded and Path(recorded).is_dir():artifact_source=Path(recorded)
    started=time.time();deadline=started+60*a.duration_minutes
    if a.compute_cutoff_epoch is not None:
        assert a.compute_cutoff_epoch>started,'Absolute compute cutoff must be in the future'
        deadline=min(deadline,a.compute_cutoff_epoch+600)
    protocol=dict(start_epoch=started,deadline_epoch=deadline,compute_cutoff_epoch=deadline-600,gpu_index=a.gpu,gpu_uuid=gpu,source_study=str(artifact_source),replay_data_source=str(source),tasks=[a.task],preserve_full_inputs=True,output_generation_cap=256,scope='One selected recipe replay; accuracy_dev only; no new method search')
    write(output/'configs/protocol.json',protocol);write(output/'configs/run_deadline.json',dict(deadline_epoch=deadline))
    write(output/'results/active_gpu.json',dict(gpu_index=a.gpu,gpu_uuid=gpu))
    write(output/'configs'/f'{a.task}_warmup.json',warm);write(output/'configs/selected_job.json',[job])
    statistical=a.module in ['online_ridge','online_ridge_v2','memory_cv'];pseed=0 if statistical else a.predictor_seed
    commands=[[sys.executable,str(output/'scripts'/module_file.name),'--task',a.task,'--jobs',str(output/'configs/selected_job.json'),'--seed',str(a.seed)]]
    if not statistical:commands[0]+= ['--predictor-seed',str(pseed)]
    if a.evaluate:
        entries=output/'configs/selected_entries.json'
        commands.append([sys.executable,str(output/'scripts/make_entries.py'),'--task',a.task,'--jobs',str(output/'configs/selected_job.json'),'--seed',str(a.seed),'--predictor-seed',str(pseed),'--output',str(entries)])
        commands.append([sys.executable,str(output/'scripts/evaluate.py'),'--model','1p5b','--entries',str(entries),'--name','selected_replay','--split','accuracy_dev','--max-new-tokens','256'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),BENCH_RUN_ID=output.name,BENCH_FINAL_VALIDATION='1',PREDICTOR_MODEL_PATH=str(model),PYTHONPATH=str(output/'scripts'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    receipt=dict(prepared=time.time(),commands=commands,source_jobs_sha256=digest(a.jobs),warmup_sha256=warm['adapter_sha256'],full_inputs=True,task=a.task,recipe=a.recipe,steps=job['steps'],update_seed=a.seed,predictor_seed=pseed,compute_cutoff_epoch=deadline-600,prepare_only=a.prepare_only,evaluation_split='accuracy_dev' if a.evaluate else None,requirement='Start on an otherwise vacant chosen GPU. The trainer refuses pre-existing GPU processes and holds unused VRAM in its allocator; this is not an administrator-enforced exclusive reservation.')
    write(output/'replay_receipt.json',receipt)
    print(json.dumps(dict(output=str(output),prepare_only=a.prepare_only,commands=commands)),flush=True)
    if a.prepare_only:return
    import fcntl
    admission_lock=open(f'/tmp/h200_gpu_admission_{a.gpu}.lock','a+')
    fcntl.flock(admission_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for index,command in enumerate(commands):
        with (output/'logs'/f'replay_{index}.log').open('w') as log:
            remaining=deadline-600-time.time()
            assert remaining>0,'Replay time budget exhausted'
            child=subprocess.Popen(command,env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=remaining)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if code:raise subprocess.CalledProcessError(code,command)
        if index==0:
            run=output/'models'/f'{a.task}__{job["name"]}_u{a.seed}_p{pseed}'
            summary=json.loads((run/'summary.json').read_text())
            history=json.loads((run/'history.json').read_text())
            assert summary.get('complete') and history and history[-1]['step']==job['steps'], 'Training ended before the requested horizon; partial artifacts are preserved, but this replay is not complete'
    receipt.update(completed=time.time());write(output/'replay_receipt.json',receipt)
    print(json.dumps(dict(completed=True,results=str(output/'results'))),flush=True)
    admission_lock.close()

if __name__=='__main__':main()
