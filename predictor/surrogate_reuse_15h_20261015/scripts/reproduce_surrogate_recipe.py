"""Relocate one published reuse_train receipt into a fresh replay directory.

Requires matching warmup/predictor artifacts and reconstructed full input data.
--prepare-only performs hash checks and writes commands without using a GPU.
This does not recreate an unavailable predictor checkpoint from its hash.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,hashlib,signal

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipts',type=Path,required=True);ap.add_argument('--tag',required=True)
    ap.add_argument('--warmup-adapter',type=Path,required=True);ap.add_argument('--predictor-checkpoint',type=Path)
    ap.add_argument('--data-dir',type=Path,required=True,help='Reconstructed mnli directory containing named JSONL splits')
    ap.add_argument('--model-path',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--duration-minutes',type=float,default=60)
    ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    assert a.duration_minutes>=15 and a.gpu>=0
    records=json.loads(a.receipts.read_text())['rows'];matches=[r for r in records if r['entry']['tag']==a.tag];assert len(matches)==1
    record=matches[0];e=record['entry'];cfg=record['config'];job=dict(cfg['job'])
    assert e['task']=='mnli' and job['kind'] not in ['tangent','linear','oracle_class','oracle_full'],'This runner supports reuse_train recipes only'
    assert job.get('schedule','constant')=='constant' or job['steps']==e['steps'],'Do not change a scheduled horizon'
    warm=a.warmup_adapter.resolve();data_dir=a.data_dir.resolve();model=a.model_path.resolve();output=a.output.resolve()
    assert not output.exists(),'Use a fresh output directory'
    assert digest(warm/'adapter_model.safetensors')==cfg['warmup_sha256'],'Warmup differs from the evaluated run'
    assert digest(data_dir/f'{e["subset"]}.jsonl')==cfg['data_sha256'],'Training data must match without truncation or reordering'
    assert (model/'config.json').is_file() and (warm/'adapter_config.json').is_file()
    expected_predictor=cfg.get('predictor_sha256') or cfg.get('online_surrogate_sha256')
    if expected_predictor:
        assert a.predictor_checkpoint and digest(a.predictor_checkpoint)==expected_predictor,'Supply the exact recorded predictor artifact'
        job['predictor_path']=str(a.predictor_checkpoint.resolve())
    job.update(steps=e['steps'],save=[e['steps']],diagnostics=False)
    for name in ['scripts','configs','results','models','cache','logs','data','provenance']:(output/name).mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    for path in scripts.glob('*.py'):shutil.copy2(path,output/'scripts'/path.name)
    (output/'data/mnli').symlink_to(data_dir,target_is_directory=True)
    now=time.time();deadline=now+60*a.duration_minutes
    # Prepare-only must work without a CUDA driver; resolve the UUID on launch.
    gpu_uuid=None if a.prepare_only else subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    protocol=dict(start_epoch=now,deadline_epoch=deadline,compute_cutoff_epoch=deadline-600,gpu_index=a.gpu,gpu_uuid=gpu_uuid,source_study=str(output),tasks=['mnli'],preserve_full_inputs=True,output_generation_cap=256,scope='One recorded recipe replay; no recipe selection')
    write(output/'configs/protocol.json',protocol);write(output/'configs/run_deadline.json',dict(deadline_epoch=deadline))
    write(output/'configs'/f'mnli_warmup_w{e["warmup_seed"]}.json',dict(adapter=str(warm),adapter_sha256=cfg['warmup_sha256']))
    write(output/'configs/replay_jobs.json',[job])
    cmd=[sys.executable,str(output/'scripts/reuse_train.py'),'--warmup-seed',str(e['warmup_seed']),'--subset',e['subset'],'--seed',str(e['seed']),'--jobs',str(output/'configs/replay_jobs.json'),'--suffix','_reproduced','--cache-policy',cfg.get('cache_policy','all'),'--teacher-backend',cfg.get('teacher_backend','native'),'--graph-bucket',str(cfg.get('graph_bucket',64)),'--max-graphs',str(cfg.get('max_graphs',4))]
    if not cfg.get('graph_initial_audit',True):cmd+=['--skip-graph-audit']
    receipt=dict(prepared_epoch=now,source_receipts_sha256=digest(a.receipts),reference_entry=e,reference_config=cfg,command=cmd,prepare_only=a.prepare_only,model_path=str(model),full_inputs=True,scope='Requires the specified local model and matching warmup/predictor tensors. Data hashes are checked. Exact weights are a separate validation; matching software/hardware is required for bitwise reproduction.')
    write(output/'reproduction_receipt.json',receipt)
    print(json.dumps(dict(output=str(output),prepare_only=a.prepare_only,command=cmd)),flush=True)
    if a.prepare_only:return
    import fcntl
    with open(f'/tmp/h200_gpu_admission_{a.gpu}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),BENCH_RUN_ID=output.name,BENCH_EXP_DIR=str(output),BENCH_SOURCE_DIR=str(output/'scripts'),BENCH_FINAL_VALIDATION='1',PREDICTOR_MODEL_PATH=str(model),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (output/'logs/replay.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=output,start_new_session=True)
            try:code=child.wait(timeout=max(1,deadline-600-time.time()))
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if code:raise subprocess.CalledProcessError(code,cmd)
    run=output/'models'/f'mnli__w{e["warmup_seed"]}_{e["subset"]}_{job["name"]}_u{e["seed"]}_reproduced'
    summary=json.loads((run/'summary.json').read_text());assert summary['complete']
    weights=run/f'step-{e["steps"]}'/'adapter_model.safetensors'
    receipt.update(completed_epoch=time.time(),actual_adapter_sha256=digest(weights),serialized_adapter_hash_matches=digest(weights)==e['adapter_sha256'])
    write(output/'reproduction_receipt.json',receipt);print(json.dumps(dict(complete=True,serialized_adapter_hash_matches=receipt['serialized_adapter_hash_matches'])),flush=True)

if __name__=='__main__':main()
