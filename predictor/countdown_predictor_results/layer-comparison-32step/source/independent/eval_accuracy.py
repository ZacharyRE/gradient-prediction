"""Native vLLM LoRA evaluation: read one shared base, never save merged models."""
from core import *
import os,argparse

def parallel_evaluate(a, gpus):
    """Partition adapters, not questions; preserve each full evaluation batch."""
    import subprocess, shutil
    from concurrent.futures import ThreadPoolExecutor
    root=EXP/'results/accuracy'/a.name
    root.mkdir(parents=True,exist_ok=True)
    groups=[a.adapters[i::len(gpus)] for i in range(len(gpus))]
    write(root/'parallel_dispatch.json',dict(gpus=gpus,adapters=a.adapters,files=a.files,max_new_tokens=a.max_new_tokens))
    def worker(i):
        name=a.name+f'_shard{i}'
        dest=EXP/'results/accuracy'/name
        cmd=[sys.executable,__file__,'--name',name,'--adapters',*groups[i],
             '--files',*a.files,'--max-new-tokens',str(a.max_new_tokens)]
        if a.base and i==0:cmd+=['--base']
        if dest.exists():cmd+=['--resume']
        resources=json.loads((EXP/'configs/resources.json').read_text())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[i]),PREDICTOR_EVAL_WORKER='1',
                 PREDICTOR_EVAL_MAX_SEQS=str(resources.get('max_num_seqs',128)),
                 PREDICTOR_EVAL_KV_GIB=str(resources.get('kv_cache_gib',4)))
        with (EXP/'logs'/f'{name}.log').open('a') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return dest
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        destinations=list(pool.map(worker,range(len(gpus))))
    combined={};protocols=[]
    for dest in destinations:
        protocol=json.loads((dest/'protocol.json').read_text());protocols.append(protocol)
        result=json.loads((dest/'summary.json').read_text())
        assert not (set(combined)&set(result)), 'Duplicate adapter/dataset evaluation'
        for k in result:shutil.copyfile(dest/f'{k}.jsonl',root/f'{k}.jsonl')
        combined.update(result)
    expected={Path(adapter).parent.name+'__'+Path(adapter).name+'__'+Path(f).stem for adapter in a.adapters for f in a.files}
    if a.base:expected.update('base__'+Path(f).stem for f in a.files)
    assert set(combined)==expected
    assert all(p['input_sha256']==protocols[0]['input_sha256'] for p in protocols)
    write(root/'protocol.json',dict(args=vars(a),parallel_gpus=gpus,shards=protocols,
        partition='adapters only; complete dataset evaluated on each adapter',input_truncation=False))
    write(root/'summary.json',combined)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--adapters',nargs='*',default=[])
    ap.add_argument('--files',nargs='+',default=[str(data_path('accuracy_dev256'))]);ap.add_argument('--max-new-tokens',type=int,default=2048)
    ap.add_argument('--base',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    existing=EXP/'results/accuracy'/a.name
    if (existing/'summary.json').exists() and (existing/'protocol.json').exists():
        protocol=json.loads((existing/'protocol.json').read_text())
        if 'shards' in protocol:
            expected={Path(ad).parent.name+'__'+Path(ad).name+'__'+Path(f).stem for ad in a.adapters for f in a.files}
            if a.base:expected.update('base__'+Path(f).stem for f in a.files)
            summary=json.loads((existing/'summary.json').read_text())
            assert set(summary)==expected
            ih={str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.files}
            ah={str(ad):hashlib.sha256((Path(ad)/'adapter_model.safetensors').read_bytes()+(Path(ad)/'adapter_config.json').read_bytes()).hexdigest() for ad in a.adapters}
            saved={}
            for shard in protocol['shards']:
                assert shard['input_sha256']==ih and shard['max_new_tokens']==a.max_new_tokens
                saved.update(shard['adapter_sha256'])
            assert saved==ah
            assert all((existing/(key+'.jsonl')).exists() for key in summary)
            print('Validated completed parallel evaluation; reusing full outputs.',flush=True);return
    resource=EXP/'configs/resources.json'
    if resource.exists() and os.environ.get('PREDICTOR_EVAL_WORKER')!='1':
        gpus=json.loads(resource.read_text()).get('evaluation_gpus',[])
        if os.environ.get('PREDICTOR_EVAL_GPUS'):gpus=[int(x) for x in os.environ['PREDICTOR_EVAL_GPUS'].split(',')]
        dispatch=existing/'parallel_dispatch.json'
        if dispatch.exists():
            recorded=json.loads(dispatch.read_text())
            assert recorded['adapters']==a.adapters and recorded['files']==a.files and recorded['max_new_tokens']==a.max_new_tokens
            gpus=recorded['gpus']
        if len(gpus)>1 and len(a.adapters)>=len(gpus):
            parallel_evaluate(a,gpus);return
    os.environ['VLLM_BATCH_INVARIANT']='1'
    max_seqs=int(os.environ.get('PREDICTOR_EVAL_MAX_SEQS','128'))
    cache_bytes=int(os.environ.get('PREDICTOR_EVAL_KV_GIB','4'))*1024**3
    from transformers import AutoTokenizer,AutoConfig
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    tok=AutoTokenizer.from_pretrained(MODEL);datasets={Path(f).stem:read(f) for f in a.files}
    prompts={n:[prompt(tok,r) for r in rs] for n,rs in datasets.items()}
    lens=[len(tok.encode(p,add_special_tokens=False)) for ps in prompts.values() for p in ps]
    maxlen=((max(lens)+a.max_new_tokens+255)//256)*256
    assert maxlen<=AutoConfig.from_pretrained(MODEL).max_position_embeddings
    root=EXP/'results/accuracy'/a.name
    input_hashes={str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in a.files}
    adapter_hashes={str(f):hashlib.sha256((Path(f)/'adapter_model.safetensors').read_bytes()+
        (Path(f)/'adapter_config.json').read_bytes()).hexdigest() for f in a.adapters}
    protocol=dict(args=vars(a),base=MODEL,backend='native vLLM BF16 LoRA',input_sha256=input_hashes,adapter_sha256=adapter_hashes,
        input_truncation=False,max_prompt_tokens=max(lens),max_new_tokens=a.max_new_tokens,
        batch_invariant=True,temperature=0,seed=42,max_num_seqs=max_seqs,kv_cache_memory_bytes=cache_bytes,gpu_memory_startup_fraction=.08,checker=str(OLD/'scripts/countdown.py'))
    if a.resume:
        old=json.loads((root/'protocol.json').read_text())
        assert {k:v for k,v in old['args'].items() if k!='resume'}=={k:v for k,v in vars(a).items() if k!='resume'}
        assert old['input_sha256']==input_hashes and old['adapter_sha256']==adapter_hashes
        assert old.get('max_num_seqs',128)==max_seqs and old['kv_cache_memory_bytes']==cache_bytes
    else:
        root.mkdir(parents=True,exist_ok=False);write(root/'protocol.json',protocol)
    summaries=json.loads((root/'summary.json').read_text()) if a.resume and (root/'summary.json').exists() else {}
    models=([None] if a.base else [])+a.adapters
    expected=[('base' if adapter is None else Path(adapter).parent.name+'__'+Path(adapter).name)+'__'+dataset
        for adapter in models for dataset in datasets]
    if a.resume and all(k in summaries for k in expected):
        print('All requested evaluations are already complete; validated input and adapter hashes.',flush=True);return
    llm=LLM(model=MODEL,dtype='bfloat16',max_model_len=maxlen,max_num_seqs=max_seqs,
        kv_cache_memory_bytes=cache_bytes,gpu_memory_utilization=.08,seed=42,
        enable_lora=bool(a.adapters),max_lora_rank=64,max_loras=1,
        enable_prefix_caching=False,async_scheduling=False,enforce_eager=True,
        attention_config={'backend':'TRITON_ATTN'})
    try:
        for j,adapter in enumerate(models):
            name='base' if adapter is None else Path(adapter).parent.name+'__'+Path(adapter).name
            req=None if adapter is None else LoRARequest(name,j+1,str(Path(adapter).resolve()))
            for dataset,rs in datasets.items():
                if name+'__'+dataset in summaries:continue
                start=time.time();out=llm.generate(prompts[dataset],SamplingParams(temperature=0,max_tokens=a.max_new_tokens,seed=42),lora_request=req)
                preds=[]
                for r,pr,o in zip(rs,prompts[dataset],out):
                    g=o.outputs[0];verdict=check(g.text,r['nums'],r['target'])
                    preds.append(dict(source_index=r['source_index'],nums=r['nums'],target=r['target'],
                        full_prompt=pr,prediction=g.text,tokens=len(g.token_ids),finish_reason=g.finish_reason,**verdict))
                jsonl(root/f'{name}__{dataset}.jsonl',preds)
                summaries[name+'__'+dataset]=dict(n=len(preds),correct=sum(r['correct'] for r in preds),
                    accuracy=sum(r['correct'] for r in preds)/len(preds),
                    limit_hits=sum(r['finish_reason']=='length' for r in preds),seconds=time.time()-start)
                write(root/'summary.json',summaries);print(json.dumps(summaries),flush=True)
    finally:llm.llm_engine.engine_core.shutdown(timeout=15)
if __name__=='__main__':main()
