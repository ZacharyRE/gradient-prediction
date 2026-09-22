"""Durable stage runner. Refuse partial training directories; never overwrite them."""
import os,sys,json,time,subprocess,traceback,fcntl,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
EXP=Path(__file__).resolve().parents[1]
GPUS=[0,1]
CFG=json.loads((EXP/'configs/experiment.json').read_text())
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
BASE_ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PREDICTOR_EVAL_MAX_SEQS='512',PREDICTOR_EVAL_KV_GIB='16')

def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def status(stage,**kwargs):
    row=dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),gpus=GPUS,**kwargs)
    write(EXP/'status.json',row);print(json.dumps(row),flush=True)

def run(label,script,args,gpu,done):
    if done.exists():
        record=json.loads(done.read_text());assert record.get('complete') is True,(label,record)
        print('Already complete: '+label,flush=True);return
    env=dict(BASE_ENV,CUDA_VISIBLE_DEVICES=str(gpu))
    cmd=[sys.executable,'-u',str(EXP/'scripts'/script),*map(str,args)]
    print(json.dumps(dict(start=label,gpu=gpu,command=cmd)),flush=True)
    with (EXP/'logs'/f'{label}.log').open('a') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert done.exists(),(label,'completion artifact missing')
    assert json.loads(done.read_text()).get('complete') is True,label

def pool_stage(jobs):
    # Each worker holds one physical GPU and runs its list sequentially.
    def worker(gpu,jobs):
        for label,script,args,done in jobs:run(label,script,args,gpu,done)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures=[pool.submit(worker,g,jobs[i::len(GPUS)]) for i,g in enumerate(GPUS)]
        for f in futures:f.result()

def evaluate(name,adapters,dataset):
    cmd=[sys.executable,'-u',str(EXP/'scripts/eval_accuracy.py'),'--name',name,'--adapters',*map(str,adapters),'--files',str(EXP/'data'/f'{dataset}.jsonl'),'--max-new-tokens','2048']
    if (EXP/'results/accuracy'/name).exists():cmd+=['--resume']
    with (EXP/'logs'/f'{name}.log').open('a') as log:subprocess.run(cmd,env=BASE_ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((EXP/'results/accuracy'/name/'summary.json').read_text())

def main():
    lock=(EXP/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('warmup')
    run('warmup','warmup.py',[],0,EXP/'models/teacher/audit.json')
    status('collect_predictor_data')
    run('collect_joint','collect_joint.py',[],0,EXP/'results/collection.json')
    status('offline_predictor_training',max_epochs=100,lr=3e-4)
    jobs=[]
    for rep in CFG['replicates']:
        for l in CFG['layers']:
            name=f"independent_l{l:02d}_s{rep['train_seed']}"
            jobs.append((name,'fit_ablation.py',['--name',name,'--cache',EXP/'cache'/f'layer{l:02d}','--source','y','--aux','mask_pos','--seed',rep['train_seed'],'--epochs',100,'--preload'],EXP/'models'/name/'summary.json'))
    pool_stage(jobs)
    # Retain metadata/manifests; remove reproducible large shards after all nine fits succeed.
    cleanup=EXP/'results/cache_cleanup.json'
    if not cleanup.exists():
        records=[]
        for p in sorted((EXP/'cache').glob('layer*/predictor_*/shard*.pt')):
            records.append(dict(path=str(p.relative_to(EXP)),bytes=p.stat().st_size));p.unlink()
        write(cleanup,dict(complete=True,deleted=records,total_bytes=sum(r['bytes'] for r in records)))
    status('joint_lora_updates',steps=32,calibration_lr=1e-4)
    jobs=[]
    for rep in CFG['replicates']:
        for mode in CFG['methods']:
            name=f"three_{mode}_s{rep['train_seed']}_u{rep['update_seed']}"
            jobs.append((name,'update_adaptive.py',['--scope','three','--mode',mode,'--train-seed',rep['train_seed'],'--seed',rep['update_seed'],'--name',name,'--steps',32],EXP/'models'/name/'audit.json'))
    pool_stage(jobs)
    results={}
    for split,steps,dataset in [('dev',CFG['dev_steps_single'],'accuracy_dev256'),('test',CFG['test_steps_single'],'countdown_test2048')]:
        status('evaluate_'+split)
        adapters=[EXP/'models/teacher/step-32']
        for rep in CFG['replicates']:
            for mode in CFG['methods']:
                adapters.extend(EXP/'models'/f"three_{mode}_s{rep['train_seed']}_u{rep['update_seed']}"/f'step-{step}' for step in steps)
        results[split]=evaluate('three_'+split,adapters,dataset)
        rows=[json.loads(line) for line in (EXP/'data'/f'{dataset}.jsonl').read_text().splitlines()]
        assert len(results[split])==len(adapters)
        prompt_reference=None
        for key,summ in results[split].items():
            predictions=[json.loads(line) for line in (EXP/'results/accuracy'/('three_'+split)/(key+'.jsonl')).read_text().splitlines()]
            assert len(predictions)==len(rows)==summ['n']
            assert [r['source_index'] for r in rows]==[r['source_index'] for r in predictions]
            assert sum(p['correct'] for p in predictions)==summ['correct']
            prompts=[p['full_prompt'] for p in predictions]
            if prompt_reference is None:prompt_reference=prompts
            assert prompts==prompt_reference
    write(EXP/'results/summary.json',dict(complete=True,accuracy=results,input_truncation=False,dev_adapter_count=73,test_adapter_count=37))
    status('complete',complete=True)

if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc());raise
