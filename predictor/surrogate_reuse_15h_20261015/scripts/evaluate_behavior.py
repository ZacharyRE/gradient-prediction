"""Compare a small behavior readout with exact teacher forced-format ABC logits.

Every model input uses the SAME generated answer template, regardless of gold.
This is a classification proxy; generated-answer accuracy is evaluated separately.
"""
from reuse_train import *
from behavior_surrogate import BehaviorSurrogate,AnchoredBehaviorSurrogate
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel,SDPBackend

def fixed_template_dataset(task,split,tok):
    ds=Dataset(task,split,tok);answer=tok.encode('\\boxed{A}',add_special_tokens=False)+[tok.eos_token_id]
    for r,source_row in zip(ds.items,ds.rows):
        length=next(i for i,v in enumerate(r['labels']) if v!=-100)
        prefix=r['input_ids'][:length]
        assert prefix==tok.encode(prompt(tok,source_row),add_special_tokens=False),'Refuse altered or truncated original input'
        r.update(input_ids=prefix+answer,labels=[-100]*length+answer,n=len(answer))
    ds.lengths=[len(r['input_ids']) for r in ds.items]
    return ds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--entries',required=True);ap.add_argument('--predictors',nargs='+',required=True);ap.add_argument('--split',default='accuracy_dev');ap.add_argument('--name',required=True);ap.add_argument('--teacher',action='store_true');a=ap.parse_args()
    entries=json.loads(Path(a.entries).read_text());assert entries and {e['task'] for e in entries}=={'mnli'}
    out=EXP/'results/behavior'/a.name;out.mkdir(parents=True,exist_ok=True)
    freeze=None
    if a.split in ['selection_dev','selection_holdout']:
        from surrogate_freeze import ensure_surrogate_freeze,FREEZE
        from types import SimpleNamespace
        freeze=ensure_surrogate_freeze(SimpleNamespace(model='1p5b',max_new_tokens=256,adapter_group_size=1,split=a.split),entries)
        frozen=json.loads(FREEZE.read_text());declared=frozen['behavior_predictors']
        assert frozen['behavior_code']=={name:sha(CODE_ROOT/name) for name in ['evaluate_behavior.py','behavior_surrogate.py']}
        for path in a.predictors:assert declared[str(Path(path).resolve())]==sha(Path(path))
    started=time.time();m,tok,mod=setup('1p5b',entries[0]['adapter'],rng=881);A,B,s=weights(mod);cap=Capture(mod);ds=fixed_template_dataset('mnli',a.split,tok);configure(True,8192)
    cache=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32);tick=clock();cached=cached_rows(cache,ds)
    for r in cached:r['want_h']=True
    metadata_seconds=clock()-tick;models=[]
    for path in a.predictors:
        obj=torch.load(path,map_location='cuda',weights_only=False);constructor=AnchoredBehaviorSurrogate if obj['config'].get('kind')=='anchored_behavior' else BehaviorSurrogate;p=constructor(**obj['config']).cuda();p.load_state_dict(obj['state_dict']);p.eval();models.append((Path(path).parent.name,p))
    class_tokens=models[0][1].class_tokens;head=m.get_base_model().lm_head.weight[class_tokens].detach();gold=torch.tensor(['ABC'.index(r['gold']) for r in ds.rows],device='cuda')
    def exact_class_logits(ids,d):
        bb=pc.batch(ds,ids,tok)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
            hidden=m.get_base_model().model(input_ids=bb['input_ids'],attention_mask=bb['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
            where=(d['label_ids'][...,None]==class_tokens[None,None,:]).any(-1).long().argmax(1)
            logits=hidden[torch.arange(len(ids),device='cuda'),where]@head.T
        return logits.float()
    def load_adapter(path):
        vals=load_file(str(Path(path)/'adapter_model.safetensors'))
        with torch.no_grad():A.copy_(next(v for k,v in vals.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in vals.items() if k.endswith('lora_B.weight')))
    anchors={};anchor_receipts={};anchored=[p for _,p in models if getattr(p,'requires_initial_teacher',False)]
    if anchored:
        for warm in sorted({e['warmup_seed'] for e in entries}):
            tick=clock();initial=json.loads((EXP/'configs'/f'mnli_warmup_w{warm}.json').read_text());initial_sha=sha(Path(initial['adapter'])/'adapter_model.safetensors');assert initial_sha==initial['adapter_sha256']
            if freeze:assert any(r['entry'].get('steps')==0 and r['entry'].get('warmup_seed')==warm and r['adapter_sha256']==initial_sha for r in frozen['entries'].values()),'Initial anchor must also be a frozen warmup adapter'
            load_adapter(initial['adapter']);features=[];logits=[]
            for ids in chunks(ds,list(range(len(ds))),max_n=32,token_budget=8192):
                d=pack(current_features(mod,cached,ids))
                with torch.no_grad():features.append(anchored[0].features(d,A,B,s).detach())
                logits.append(exact_class_logits(ids,d).detach())
            yy=torch.cat(logits);yy-=yy.mean(-1,keepdim=True);anchors[warm]=dict(features=torch.cat(features),logits=yy);seconds=clock()-tick
            anchor_receipts[str(warm)]=dict(adapter=initial['adapter'],adapter_sha256=initial_sha,seconds=seconds,accuracy=float((yy.argmax(-1)==gold).float().mean()),scope='Paid initial-agent forced-format readout and features, once per warmup and evaluation population; reused across candidate adapters/models. A no-change selector ties all candidates.')
            jsonl(out/f'initial_teacher_w{warm}.jsonl',[dict(sample_id=row['sample_id'],gold=row['gold'],prediction='ABC'[int(logit.argmax())],logits=logit.tolist(),correct=bool(logit.argmax()==target)) for row,target,logit in zip(ds.rows,gold.cpu(),yy.cpu())])
    reserve_worker_cache();protocol=dict(args=vars(a),entries_sha256=sha(Path(a.entries)),predictor_sha256={p:sha(Path(p)) for p in a.predictors},data_sha256=sha(data('mnli',a.split)),adapter_sha256={e['adapter']:sha(Path(e['adapter'])/'adapter_model.safetensors') for e in entries},source_sha256=source_signature(__file__),full_original_prompts=True,fixed_answer_template='\\boxed{A}',gold_class_never_enters_model_input=True,target='ABC next-token distribution under fixed boxed-answer prefix; NOT free generation accuracy',cache_seconds=cache.seconds,metadata_seconds=metadata_seconds,freeze=freeze)
    protocol['initial_teacher_anchors']=anchor_receipts
    assert not (out/'protocol.json').exists(),'Use a fresh name to preserve timing/protocol receipts';write(out/'protocol.json',protocol);summaries={}
    for e in entries:
        if deadline(900):raise RuntimeError('Compute deadline')
        tick=clock();load_adapter(e['adapter'])
        load_seconds=clock()-tick;preds={name:[] for name,_ in models};teacher=[];times={name:0. for name,_ in models};feature_seconds=0.;teacher_seconds=0.
        for ids in chunks(ds,list(range(len(ds))),max_n=32,token_budget=8192):
            tick=clock();rows=current_features(mod,cached,ids);d=pack(rows);feature_seconds+=clock()-tick
            for name,p in models:
                tick=clock()
                with torch.no_grad():
                    features=p.features(d,A,B,s)
                    if getattr(p,'requires_initial_teacher',False):
                        anchor=anchors[e['warmup_seed']];logits=p(features,anchor['features'][ids],anchor['logits'][ids])
                    else:logits=p(features)
                times[name]+=clock()-tick;preds[name].append(logits.detach().cpu())
            if a.teacher:
                tick=clock();logits=exact_class_logits(ids,d)
                teacher_seconds+=clock()-tick;teacher.append(logits.float().cpu())
        teacher_logits=torch.cat(teacher) if teacher else None;g=gold.cpu();report={}
        for name,parts in preds.items():
            logits=torch.cat(parts);labels=logits.argmax(-1);r=dict(n=len(ds),accuracy=float((labels==g).float().mean()),mean_gold_probability=float(logits.softmax(-1)[torch.arange(len(ds)),g].mean()),seconds=feature_seconds+times[name],current_hidden_seconds=feature_seconds,small_model_seconds=times[name],adapter_load_seconds=load_seconds,one_time_cache_seconds=cache.seconds+metadata_seconds)
            if e.get('steps')==0 and getattr(dict(models)[name],'requires_initial_teacher',False):assert torch.equal(logits,anchors[e['warmup_seed']]['logits'].cpu()),'Anchored readout must exactly retain its initial teacher'
            if teacher_logits is not None:r.update(teacher_class_agreement=float((labels==teacher_logits.argmax(-1)).float().mean()),teacher_forced_accuracy=float((teacher_logits.argmax(-1)==g).float().mean()),teacher_forced_seconds=teacher_seconds)
            if e['warmup_seed'] in anchors:
                yy=anchors[e['warmup_seed']]['logits'].cpu();r.update(initial_teacher_accuracy=float((yy.argmax(-1)==g).float().mean()),initial_teacher_class_agreement=float((labels==yy.argmax(-1)).float().mean()),one_time_initial_teacher_seconds=anchor_receipts[str(e['warmup_seed'])]['seconds'])
                if teacher_logits is not None:
                    target=teacher_logits-teacher_logits.mean(-1,keepdim=True);prediction=logits-logits.mean(-1,keepdim=True)
                    r.update(logit_mse=float((prediction-target).square().mean()),initial_teacher_logit_mse=float((yy-target).square().mean()),initial_teacher_vs_current_teacher_agreement=float((yy.argmax(-1)==target.argmax(-1)).float().mean()))
            report[name]=r
            jsonl(out/f'{e["tag"]}__{name}.jsonl',[dict(sample_id=row['sample_id'],gold=row['gold'],prediction='ABC'[int(label)],logits=logit.tolist(),correct=bool(label==target)) for row,label,target,logit in zip(ds.rows,labels,g,logits)])
        if teacher_logits is not None:jsonl(out/f'{e["tag"]}__teacher.jsonl',[dict(sample_id=row['sample_id'],gold=row['gold'],prediction='ABC'[int(logit.argmax())],logits=logit.tolist(),correct=bool(logit.argmax()==target)) for row,target,logit in zip(ds.rows,g,teacher_logits)])
        summaries[e['tag']]=report;write(out/'summary.json',summaries);log(dict(tag=e['tag'],models=report))
    cache.close();cap.close();write(out/'complete.json',dict(complete=True,wall_seconds=time.time()-started,entries=len(entries)))

if __name__=='__main__':main()
