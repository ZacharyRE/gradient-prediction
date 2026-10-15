"""Per-training-example gradient memory as a cheap finite-dataset surrogate.

The table is fixed before correction sampling and refreshed afterward. This is
not a neural predictor and cannot generalize to unseen examples. Snapshot
initialization costs and examples are explicitly charged. Uniform CV correction
remains unbiased conditional on the current batch, before clipping and Adam.
"""
from online_ridge import *
ENTRY_SHA=sha(Path(__file__))
CODE_SHA=source_signature(Path(__file__))

def run_memory(task,jobs,seed_id):
    snapshot=EXP/'provenance/source_versions'/CODE_SHA;snapshot.mkdir(parents=True,exist_ok=True)
    for script in (EXP/'scripts').glob('*.py'):
        if not (snapshot/script.name).exists():shutil.copy2(script,snapshot/script.name)
    warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=seed_id);A,B,s=weights(mod);original=[v.detach().clone() for v in [A,B]];basever=versions(m);cap=Capture(mod)
    ds=Dataset(task,'update',tok);dev=Dataset(task,'accuracy_dev',tok)
    configure(True,8192);initial=None;initial_seconds=0.;entries=[]
    classes={v:i for i,v in enumerate(sorted(set(r['gold'] for r in ds.rows)))}
    classids=torch.tensor([classes[r['gold']] for r in ds.rows],device='cuda')
    for job in jobs:
        name=f'{task}__{job["name"]}_u{seed_id}_p0';out=EXP/'models'/name
        if (out/'summary.json').exists():entries.extend(json.loads((out/'summary.json').read_text())['entries']);continue
        if deadline(5400):raise RuntimeError('Final validation reserve reached')
        out.mkdir(exist_ok=True);seed(seed_id)
        with torch.no_grad():A.copy_(original[0]);B.copy_(original[1])
        dim=A.numel()+B.numel()
        if job.get('initialize') and initial is None:
            started=clock();initial=torch.zeros(len(ds),dim,device='cuda')
            for ids in chunks(ds,list(range(len(ds))),max_n=32):
                rows,_,_=collect(m,tok,mod,cap,ds,ids,micro=32,keep=True,verify_reconstruction=False)
                initial[ids]=individual_truth(rows,A,B,s)
            initial_seconds=clock()-started
        table=initial.clone() if job.get('initialize') else torch.zeros(len(ds),dim,device='cuda')
        seen=torch.full((len(ds),),bool(job.get('initialize')),device='cuda',dtype=torch.bool)
        ridge=OnlineRidge(len(classes),dim,job.get('ridge',10.),job.get('forget',.99))
        phi=F.one_hot(classids,len(classes)).float()
        reserve_worker_cache()
        opt=torch.optim.AdamW([A,B],lr=job['lr'],weight_decay=0)
        rng=np.random.default_rng(seed_id);order=[]
        while len(order)<job['steps']*32:order.extend(map(int,rng.permutation(len(ds))))
        crng=np.random.default_rng(seed_id+10000);history=[];ee=[];started=time.time()
        write(out/'config.json',dict(task=task,job=job,update_seed=seed_id,predictor_seed=0,code_sha256=ENTRY_SHA,source_bundle_sha256=CODE_SHA,warmup_sha256=warm['adapter_sha256'],full_inputs=True,gradient_memory_parameters=table.numel(),prediction_before_current_labels=True,limitation='Finite training-example table, no unseen-example generalization; no neural predictor'))
        for st in range(1,job['steps']+1):
            if deadline(3600):raise RuntimeError('Final hour reserve')
            ids=order[(st-1)*32:st*32];positions=list(map(int,crng.choice(32,job.get('k',4),replace=False)));subset=[ids[i] for i in positions]
            tick=clock();pred=table[ids].clone()
            if job.get('class_fallback'):
                fallback=ridge.predict(phi[ids]);pred=torch.where(seen[ids,None],pred,fallback)
            c=job.get('coefficient',1.);n=sum(ds[i]['n'] for i in ids)
            t=clock();rows,truth,_=collect(m,tok,mod,cap,ds,subset,micro=job.get('k',4),keep=True,verify_reconstruction=False);label_seconds=clock()-t
            with torch.no_grad():
                gradients=(flat(truth)*(32/len(positions))+c*(pred.sum(0)-(32/len(positions))*pred[positions].sum(0)))/n
                targets=individual_truth(rows,A,B,s)
                pc=pred[positions]-pred[positions].mean(0);tc=targets-targets.mean(0)
                residual_ratio=float((tc-c*pc).square().sum()/tc.square().sum().clamp_min(1e-20))
            opt.zero_grad(set_to_none=True);A.grad=gradients[:A.numel()].reshape_as(A);B.grad=gradients[A.numel():].reshape_as(B)
            norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
            t=clock()
            with torch.no_grad():
                if job.get('refresh_memory',True):table[subset]=targets;seen[subset]=True
                if job.get('class_fallback'):ridge.update(phi[subset],targets)
            cal_seconds=clock()-t
            row=dict(step=st,online_seconds=clock()-tick,label_seconds=label_seconds,calibration_seconds=cal_seconds,true_gradient_examples=len(subset),gradient_norm=norm,indices=ids,positions=positions,correction_sample_residual_ratio=residual_ratio)
            history.append(row)
            if st in job.get('save',[32,64,128,256,512,1024]):
                dest=out/f'step-{st}';m.save_pretrained(dest)
                t=clock();row['dev_ce']=ce(m,tok,dev,ids=list(range(128)),micro=16);row['audit_seconds']=clock()-t
                e=dict(task=task,tag=f'{name}__step{st}',adapter=str(dest),method=job['name'],lr=job['lr'],seed=seed_id,predictor_seed=0,steps=st,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=0.,offline_fit_seconds=initial_seconds if job.get('initialize') else 0.,offline_gradient_examples=len(ds) if job.get('initialize') else 0,dev_ce=row['dev_ce'],adapter_sha256=sha(dest/'adapter_model.safetensors'))
                ee.append(e);write(out/'entries.json',ee)
            if st%32==0 or st==1:
                write(out/'history.json',history);log(dict(run=name,step=st,online_seconds=sum(r['online_seconds'] for r in history),dev_ce=row.get('dev_ce')))
        assert versions(m)==basever
        write(out/'history.json',history);write(out/'summary.json',dict(complete=True,entries=ee,online_seconds=sum(r['online_seconds'] for r in history),wall_seconds=time.time()-started,cache_seconds=0.,offline_fit_seconds=initial_seconds if job.get('initialize') else 0.,offline_gradient_examples=len(ds) if job.get('initialize') else 0,true_gradient_examples=sum(r['true_gradient_examples'] for r in history),base_unchanged=True))
        entries.extend(ee);del table,ridge,opt;gc.collect()
    cap.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);a=ap.parse_args()
    run_memory(a.task,json.loads(Path(a.jobs).read_text()),a.seed)

if __name__=='__main__':main()
