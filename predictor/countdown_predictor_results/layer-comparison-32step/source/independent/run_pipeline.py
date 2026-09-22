"""Independent layer 3 and 19 runs; reuse the completed layer 8 reference."""
import os,sys,json,subprocess,traceback,fcntl,hashlib,csv,statistics
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
EXP=Path(__file__).resolve().parents[1]
CFG=json.loads((EXP/'configs/experiment.json').read_text())
GPUS=CFG['resources']['gpus'];LAYERS=CFG['layers']
assert GPUS==[0,1] and LAYERS==[3,19]
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PREDICTOR_LAYER='3',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PREDICTOR_EVAL_MAX_SEQS='512',PREDICTOR_EVAL_KV_GIB='16')

def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def status(stage,**kwargs):
    row=dict(stage=stage,pid=os.getpid(),layers=LAYERS,reused_layer=8,gpus=GPUS,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    write(EXP/'status.json',row);print(json.dumps(row),flush=True)

def run(job,gpu):
    label,script,args,layer,done=job
    if done.exists():
        assert json.loads(done.read_text()).get('complete') is True,label
        print('Already complete: '+label,flush=True);return
    env=dict(ENV,CUDA_VISIBLE_DEVICES=str(gpu),PREDICTOR_LAYER=str(layer))
    cmd=[sys.executable,'-u',str(EXP/'scripts'/script),*map(str,args)]
    print(json.dumps(dict(start=label,layer=layer,gpu=gpu,command=cmd)),flush=True)
    with (EXP/'logs'/f'{label}.log').open('a') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert done.exists() and json.loads(done.read_text()).get('complete') is True,label

def pool_stage(jobs):
    def worker(gpu,subset):
        for job in subset:run(job,gpu)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures=[pool.submit(worker,g,jobs[i::len(GPUS)]) for i,g in enumerate(GPUS)]
        for f in futures:f.result()

def reference_audit():
    manifest=json.loads((EXP/'provenance/manifest.json').read_text())
    assert all(hashlib.sha256((EXP/path).read_bytes()).hexdigest()==h for path,h in manifest['sha256'].items())
    for split,n,k in [('dev',256,73),('test',2048,37)]:
        summary=json.loads((EXP/'provenance'/f'layer8_{split}_summary.json').read_text())
        assert len(summary)==k and all(v['n']==n for v in summary.values())
        protocol=json.loads((EXP/'provenance'/f'layer8_{split}_protocol.json').read_text())
        dataset='accuracy_dev256' if split=='dev' else 'countdown_test2048'
        checksum=hashlib.sha256((EXP/'data'/f'{dataset}.jsonl').read_bytes()).hexdigest()
        for shard in protocol['shards']:
            assert list(shard['input_sha256'].values())==[checksum]
            assert shard['max_num_seqs']==512 and shard['max_new_tokens']==2048 and shard['input_truncation'] is False
    for f,h in json.loads((EXP/'provenance/launch_code_sha256.json').read_text()).items():
        assert hashlib.sha256((EXP/f).read_bytes()).hexdigest()==h,('Code changed',f)
    write(EXP/'results/reference_audit.json',dict(complete=True,layer8_reference_complete=True,data_matches_reference=True,source_hashes_verified=True))

def warmup_audit():
    import numpy as np
    count=len((EXP/'data/warmup.jsonl').read_text().splitlines())
    for l in LAYERS:
        root=EXP/'models'/f'teacher_l{l:02d}';a=json.loads((root/'audit.json').read_text())
        assert a['complete'] and a['base_unchanged'] and a['layers']==[l]
        cfg=json.loads((root/'config.json').read_text());assert cfg['trainable_parameters']==114688
        rng=np.random.RandomState(101);order=[]
        h=json.loads((root/'history.json').read_text());assert len(h)==32
        for row in h:
            if len(order)<32:order=list(rng.permutation(count))
            ids,order=order[:32],order[32:];assert ids==row['sample_indices']
    write(EXP/'results/warmup_audit.json',dict(complete=True,exactly_one_layer_per_model=True,layers=LAYERS,original_seed101_initialization_verified=True,original_warmup_sample_order_verified=True,steps=32))

def update_audit():
    for l in LAYERS:
        for rep in CFG['replicates']:
            for mode in CFG['methods']:
                suffix=f"{mode}_s{rep['train_seed']}_u{rep['update_seed']}"
                path=EXP/'models'/f'single_l{l:02d}_{suffix}'
                audit=json.loads((path/'audit.json').read_text());assert audit['complete'] and audit['base_unchanged']
                cfg=json.loads((path/'config.json').read_text());assert cfg['layers']==[l]
                h=json.loads((path/'history.json').read_text());old=json.loads((EXP/'provenance/layer8_runs'/f'single_{suffix}'/'history.json').read_text())
                assert len(h)==len(old)==32
                for a,b in zip(h,old):
                    for key in ['sample_indices','calibration_indices','heldout_indices']:assert a[key]==b[key]
                    if mode!='oracle':assert set(a['diagnostics'])=={str(l)}
    write(EXP/'results/update_audit.json',dict(complete=True,trajectories=24,steps=32,only_selected_layer_trainable=True,same_sample_order_as_layer8=True,same_calibration_ids_as_layer8=True))

def evaluate(split,steps,dataset):
    adapters=[]
    for l in LAYERS:
        adapters.append(EXP/'models'/f'teacher_l{l:02d}'/'step-32')
        for rep in CFG['replicates']:
            for mode in CFG['methods']:
                adapters.extend(EXP/'models'/f"single_l{l:02d}_{mode}_s{rep['train_seed']}_u{rep['update_seed']}"/f'step-{step}' for step in steps)
    name='single_layers_'+split
    cmd=[sys.executable,'-u',str(EXP/'scripts/eval_accuracy.py'),'--name',name,'--adapters',*map(str,adapters),'--files',str(EXP/'data'/f'{dataset}.jsonl'),'--max-new-tokens','2048']
    result=EXP/'results/accuracy'/name
    if result.exists():cmd+=['--resume']
    with (EXP/'logs'/f'{name}.log').open('a') as log:subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((result/'summary.json').read_text())
    inputs=[json.loads(line) for line in (EXP/'data'/f'{dataset}.jsonl').read_text().splitlines()]
    assert len(summary)==len(adapters);reference=None
    for key,summ in summary.items():
        rows=[json.loads(line) for line in (result/(key+'.jsonl')).read_text().splitlines()]
        assert len(rows)==len(inputs)==summ['n']
        assert [r['source_index'] for r in rows]==[r['source_index'] for r in inputs]
        assert sum(r['correct'] for r in rows)==summ['correct']
        prompts=[r['full_prompt'] for r in rows]
        if reference is None:reference=prompts
        assert prompts==reference
    write(EXP/'results'/f'{split}_audit.json',dict(complete=True,adapters=len(adapters),questions_per_adapter=len(inputs),full_inputs=True,source_order_verified=True,correct_counts_verified=True))
    return summary

def report(accuracy):
    allrows=[];lines=['# 独立单层比较：layer 3、8、19','','Layer 3、19 为新实验；layer 8 复用原 adaptive-validation。每个模型只训练一个 o_proj LoRA，各自先完成 32 步真实梯度 warmup。表格为三个配对种子的准确率平均值，共享 warmup 基线只评测一次。','']
    for split,steps in [('test',[1,16,32]),('dev',[1,2,4,8,16,32])]:
        old=json.loads((EXP/'provenance'/f'layer8_{split}_summary.json').read_text())
        lines += [f'## {split}', '', '| layer | warmup | 方法 | '+' | '.join(f'{s} 步' for s in steps)+' |','|---|---:|---|'+'---:|'*len(steps)]
        for l in [3,8,19]:
            data=old if l==8 else accuracy[split];prefix='single' if l==8 else f'single_l{l:02d}';teacher='teacher' if l==8 else f'teacher_l{l:02d}'
            baseline=next(v['accuracy'] for k,v in data.items() if k.startswith(teacher+'__step-32__'))
            for mode in CFG['methods']:
                values=[]
                for step in steps:
                    entries=[v for k,v in sorted(data.items()) if k.startswith(prefix+'_'+mode+'_') and f'__step-{step}__' in k]
                    assert len(entries)==3
                    avg=statistics.mean(v['accuracy'] for v in entries);values.append(avg)
                    allrows.append(dict(layer=l,split=split,mode=mode,step=step,baseline=baseline,mean_accuracy=avg,gain=avg-baseline,seed_accuracies=[v['accuracy'] for v in entries],reused=(l==8)))
                lines.append(f'| {l} | {100*baseline:.2f}% | {mode} | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
        lines += ['']
    lines += ['## 指标解释','','每步梯度诊断比较同一个 LoRA 状态、同一批样本下的预测梯度与真实反传梯度。activation 和 per-example LoRA 指标逐样本计算再平均；batch LoRA 指标在聚合后计算。relative L2 = ||预测−真实|| / ||真实||；norm ratio = ||预测|| / ||真实||。','', '不同 layer 的 warmup 基线不同，因此同时报告绝对准确率和相对各自 warmup 的增益。每个 seed 重复使用同一测试集；不将这些重复观测视为独立试题。','', '本实验回答各层独立更新表现；没有检验“三层联合模型中仅一层用 predictor、其余两层用 oracle”的混合更新。','', '所有设置预先固定，未根据当前测试结果调参。测试集为此前已使用的 benchmark，不是新的独立确认集。','']
    (EXP/'results/report.md').write_text('\n'.join(lines))
    write(EXP/'results/comparison.json',dict(complete=True,rows=allrows))
    metrics=[]
    for l in [3,8,19]:
        paths=sorted((EXP/'provenance/layer8_runs').glob('single_*')) if l==8 else sorted((EXP/'models').glob(f'single_l{l:02d}_*'))
        for path in paths:
            cfg=json.loads((path/'config.json').read_text())['args']
            if cfg['mode']=='oracle':continue
            for h in json.loads((path/'history.json').read_text()):
                for stage,d in h['diagnostics'][str(l)].items():
                    for target in ['activation','lora','batch_lora']:
                        metrics.append(dict(layer=l,mode=cfg['mode'],train_seed=cfg['train_seed'],update_seed=cfg['seed'],step=h['step'],stage=stage,target=target,**d[target]))
    with (EXP/'results/step_gradient_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)

def main():
    reference_audit();status('independent_warmup')
    pool_stage([(f'warmup_l{l:02d}','warmup.py',[],l,EXP/'models'/f'teacher_l{l:02d}'/'audit.json') for l in LAYERS])
    warmup_audit();status('collect_predictor_data')
    pool_stage([(f'collect_l{l:02d}','collect_single.py',[],l,EXP/'results'/f'collection_l{l:02d}.json') for l in LAYERS])
    status('offline_predictor_training',max_epochs=100,lr=3e-4)
    jobs=[]
    for rep in CFG['replicates']:
        for l in LAYERS:
            name=f"independent_l{l:02d}_s{rep['train_seed']}"
            jobs.append((name,'fit_ablation.py',['--name',name,'--cache',EXP/'cache'/f'layer{l:02d}','--source','y','--aux','mask_pos','--seed',rep['train_seed'],'--epochs',100,'--preload'],l,EXP/'models'/name/'summary.json'))
    pool_stage(jobs)
    cleanup=EXP/'results/cache_cleanup.json'
    if not cleanup.exists():
        removed=[]
        for path in sorted((EXP/'cache').glob('layer*/predictor_*/shard*.pt')):
            removed.append(dict(path=str(path.relative_to(EXP)),bytes=path.stat().st_size));path.unlink()
        write(cleanup,dict(complete=True,removed=removed,total_bytes=sum(r['bytes'] for r in removed)))
    status('single_layer_updates',steps=32,calibration_lr=1e-4)
    jobs=[]
    for rep in CFG['replicates']:
        for mode in CFG['methods']:
            for l in LAYERS:
                name=f"single_l{l:02d}_{mode}_s{rep['train_seed']}_u{rep['update_seed']}"
                jobs.append((name,'update_adaptive.py',['--scope','single','--mode',mode,'--train-seed',rep['train_seed'],'--seed',rep['update_seed'],'--name',name,'--steps',32],l,EXP/'models'/name/'audit.json'))
    pool_stage(jobs);update_audit()
    accuracy={}
    for split,steps,dataset in [('dev',CFG['dev_steps_single'],'accuracy_dev256'),('test',CFG['test_steps_single'],'countdown_test2048')]:
        status('evaluate_'+split);accuracy[split]=evaluate(split,steps,dataset)
    report(accuracy)
    write(EXP/'results/summary.json',dict(complete=True,layers=LAYERS,reused_layer=8,accuracy=accuracy,input_truncation=False,dev_adapter_count=146,test_adapter_count=74))
    status('complete',complete=True)

if __name__=='__main__':
    lock=(EXP/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc());raise
