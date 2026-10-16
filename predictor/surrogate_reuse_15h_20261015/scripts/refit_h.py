"""Offline refits on inherited training-only gradient cache, held-out fit selection."""
from research_v3 import *
from global_gradient_objective import estimate_factor_normalizers,objective_global

@torch.no_grad()
def score(p,rows,A,B,s):
    report,pp,tt,_=diagnose(p,rows,A,B,s,return_grads=True)
    pc=pp-pp.mean(0);tc=tt-tt.mean(0)
    report['centered_cv_ratio']=float((tc-pc).square().sum()/tc.square().sum().clamp_min(1e-20))
    report['centered_correlation']=float((pc*tc).sum()/(pc.norm()*tc.norm()).clamp_min(1e-20))
    return report

def fit(task,jobs,pseed):
    gpu_guard();guard_space();seed(pseed)
    signature=source_signature(Path(__file__))
    snapshot=EXP/'provenance/source_versions'/signature;snapshot.mkdir(parents=True,exist_ok=True)
    for script in (EXP/'scripts').glob('*.py'):
        if not (snapshot/script.name).exists():shutil.copy2(script,snapshot/script.name)
    cache=EXP/'cache'/f'{task}_h'
    train=load_samples(cache/'predictor_train');dev=load_samples(cache/'predictor_dev')
    meta=torch.load(cache/'metadata.pt',map_location='cpu',weights_only=False)
    A=meta['A'].cuda();B=meta['B'].cuda();s=meta['scale']
    normalizers=estimate_factor_normalizers(train,A,B,s)
    reserve_worker_cache()
    for job in jobs:
        out=EXP/'models'/f'{task}__refit_h_{job["name"]}_p{pseed}'
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True);seed(pseed);tick=clock()
        labels=sorted({int(v) for r in train for v in torch.unique(r['label_ids']).tolist() if v>=0}) if job.get('labels') else []
        p=Predictor(B.shape[0],width=job.get('width',512),depth=job.get('depth',2),label_dim=len(labels),label_vocab=max(labels)+1 if labels else 0,input_field='h').cuda()
        if labels:
            with torch.no_grad():p.label_table[torch.tensor(labels,device='cuda'),torch.arange(len(labels),device='cuda')]=1
        p.target_scale.fill_((sum(float(r['g'].square().sum()) for r in train)/sum(r['g'].numel() for r in train))**.5)
        opt=torch.optim.AdamW(p.parameters(),lr=job.get('fit_lr',.0003),weight_decay=.01)
        rng=np.random.default_rng(pseed);history=[];best=float('inf');selected_epoch=None;steps=0
        write(out/'config.json',dict(task=task,job=job,predictor_seed=pseed,source_cache=str(cache),cache_sha256=sha(cache/'complete.json'),train_count=len(train),dev_count=len(dev),selection='minimum centered CV residual variance ratio on predictor_dev',full_inputs=True,code_sha256=sha(Path(__file__)),source_bundle_sha256=source_signature(Path(__file__))))
        for epoch in range(job.get('epochs',30)+1):
            if deadline(5400):raise RuntimeError('Final validation reserve')
            losses=[]
            if epoch:
                order=list(rng.permutation(len(train)))
                batches=list(sample_chunks(train,order,max_n=32,budget=8192))
                for ids in batches:
                    p.train();d=pack([train[i] for i in ids]);opt.zero_grad(set_to_none=True)
                    if job.get('loss')=='centered':
                        g=p(d);pg=raw(d['x'],g,A,B,s)
                        with torch.no_grad():tg=raw(d['x'],d['g'],A,B,s)
                        loss=0.
                        for x,y,key in zip(pg,tg,['factor_A','factor_B']):
                            error=x-y;error=error-error.mean(0,keepdim=True)
                            loss=loss+.5*error.square().sum((1,2)).mean()/normalizers[key]
                        loss=loss+.05*(g-d['g']).square().sum()/d['valid'].sum()/g.shape[-1]/p.target_scale.square()
                    else:loss=objective_global(p,d,A,B,s,normalizers=normalizers)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refit loss')
                    loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1;losses.append(float(loss.detach()))
            p.eval();metrics=score(p,dev,A,B,s);value=metrics['centered_cv_ratio']
            if value<best:
                best=value;selected_epoch=epoch;save_predictor(p,out/'best.pt',epoch=epoch,dev_score=value)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,dev=metrics,seconds=clock()-tick))
            write(out/'history.json',history)
            if epoch%5==0:log(dict(task=task,refit=job['name'],epoch=epoch,best_centered_cv_ratio=best,seconds=clock()-tick))
        write(out/'summary.json',dict(complete=True,best_epoch=selected_epoch,best_centered_cv_ratio=best,seconds=clock()-tick,optimizer_steps=steps,best_sha256=sha(out/'best.pt'),offline_gradient_examples=len(train)+len(dev),offline_collection_reused=True))
        del p,opt;gc.collect()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    fit(a.task,json.loads(Path(a.jobs).read_text()),a.predictor_seed)
