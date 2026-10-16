"""CV experiments: coefficients fixed before sampling, calibration delayed until after update.

All per-step labels come from training data. Accuracy development selects recipes;
held-out gradient audits never train the predictor or select its online checkpoint.
"""
from predictor_core import *
from correction4 import per_example_predictions
from updates_legacy import clock, adam_delta
import argparse, copy, gc

SOURCE=Path(json.loads((EXP/'configs/protocol.json').read_text())['source_study'])
ENTRY_SHA=sha(Path(__file__))
CODE_SHA=source_signature(Path(__file__))
from gpu_features import features_gpu,collect_gpu
from teacher_prefix_cache import TeacherPrefixCache
features=features_gpu
collect=collect_gpu

def variance_diagnostic(p,rows,A,B,s,coefficients):
    result,pp,tt,_=diagnose(p,rows,A,B,s,return_grads=True,factor_detail=True)
    cut=A.numel();metrics={}
    for name,x,y,c in [('A',pp[:,:cut],tt[:,:cut],coefficients[0]),('B',pp[:,cut:],tt[:,cut:],coefficients[1])]:
        x=x-x.mean(0);y=y-y.mean(0)
        metrics[name]=dict(centered_correlation=float((x*y).sum()/(x.norm()*y.norm()).clamp_min(1e-20)),cv_residual_variance_ratio=float((y-c*x).square().sum()/y.square().sum().clamp_min(1e-20)),coefficient=c)
    result['control_variate']=metrics
    return result

def predictor_path(task,pseed):
    suffix='_globalfit' if task=='mnli' else ''
    return SOURCE/'models'/f'{task}_predictor{suffix}_p{pseed}'/'best.pt'

def train_calibration(p,optimizer,rows,A,B,s,job,rng):
    # This is applied after the LoRA update, affects only FUTURE estimators, and
    # uses captured pre-update A/B for consistent reconstruction targets.
    from global_gradient_objective import estimate_factor_normalizers,objective_global
    tick=clock();normalizers=estimate_factor_normalizers(rows,A,B,s)
    losses=[];p.train()
    for _ in range(job.get('cal_epochs',1)):
        order=list(rng.permutation(len(rows)))
        for ix in sample_chunks(rows,order,max_n=job.get('cal_micro',16),budget=4096):
            d=pack([rows[i] for i in ix]);optimizer.zero_grad(set_to_none=True)
            condition_lora_state(p,d,A,B,s)
            if job.get('cal_loss')=='centered':
                g=p(d);pg=raw(d['x'],g,A,B,s)
                with torch.no_grad():tg=raw(d['x'],d['g'],A,B,s)
                loss=0.
                for pp,tt,key in zip(pg,tg,['factor_A','factor_B']):
                    error=pp-tt
                    error=error-error.mean(0,keepdim=True)
                    loss=loss+.5*error.square().sum((1,2)).mean()/normalizers[key]
                loss=loss+job.get('activation_weight',.05)*(g-d['g']).square().sum()/d['valid'].sum()/g.shape[-1]/p.target_scale.square()
            else:loss=objective_global(p,d,A,B,s,normalizers=normalizers)
            if not torch.isfinite(loss):raise FloatingPointError('Calibration loss nonfinite')
            loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
    p.eval();return dict(seconds=clock()-tick,examples=len(rows),optimizer_steps=len(losses),mean_loss=float(np.mean(losses)))

@torch.no_grad()
def cache_features(m,tok,mod,cap,ds):
    tick=clock();rows=features(m,tok,mod,cap,ds,list(range(len(ds))),micro=32)
    # Only X and metadata are invariant: current Y must be recomputed each step.
    out=[]
    for r in rows:
        out.append({k:(v.cuda() if torch.is_tensor(v) else v) for k,v in r.items() if k!='y'})
    return out,clock()-tick

@torch.no_grad()
def current_features(mod,cached,ids):
    assert all(cached[i] is not None for i in ids),'Prefix planning omitted a prediction input'
    result=[]
    for pos in sample_chunks(cached,ids,max_n=32,budget=8192):
        rr=[cached[i] for i in pos];L=max(len(r['x']) for r in rr)
        x=torch.zeros(len(rr),L,rr[0]['x'].shape[-1],device='cuda',dtype=torch.bfloat16)
        for j,r in enumerate(rr):x[j,:len(r['x'])]=r['x']
        with torch.autocast('cuda',dtype=torch.bfloat16):y=mod(x)
        for j,r in enumerate(rr):
            row=dict(r,y=y[j,:len(r['x'])].detach())
            if r.get('want_h',False):row['h']=(r['residual']+row['y']).bfloat16()
            result.append(row)
    assert [r['index'] for r in result]==ids
    return result

def predictor_features(p,mod,cached,ids):
    if getattr(p,'requires_current_output',True):return current_features(mod,cached,ids)
    assert p.input_field=='x' and all(cached[i] is not None for i in ids)
    # pack() uses Y only to obtain its output width; this predictor reads X and
    # labels only. No current module output or hidden-state pass is required.
    return [dict(cached[i],y=cached[i]['x']) for i in ids]

def update(m,tok,mod,cap,p,ds,ids,positions,opt,job,step,replay,calopt,calrng,lambda_state,cached):
    A,B,s=weights(mod);tick=clock();k=len(positions);n=sum(ds[i]['n'] for i in ids)
    kind=job['kind'];refresh=bool(job.get('refresh_every') and step%job['refresh_every']==0)
    row=dict(step=step,true_gradient_examples=0,calibration_seconds=0.,prefix_prediction_seconds=0.,label_seconds=0.,refresh=refresh)
    if refresh and job.get('cal_on_refresh'):
        oldA=A.detach().clone();oldB=B.detach().clone()
        t=clock();labels,truth,_=collect(m,tok,mod,cap,ds,ids,micro=job.get('micro',16),keep=True,verify_reconstruction=False)
        row['label_seconds']=clock()-t
        opt.zero_grad(set_to_none=True)
        for w,g in zip([A,B],truth):w.grad=g/n
        row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        row['calibration']=train_calibration(p,calopt,labels,oldA,oldB,s,job,calrng)
        row['calibration_seconds']=row['calibration']['seconds'];row['true_gradient_examples']=len(ids)
        row['online_seconds']=clock()-tick
        return row
    if kind=='oracle' or refresh:
        row.update(oracle_step(m,tok,mod,ds,ids,opt,micro=job.get('micro',4)))
        row['true_gradient_examples']=len(ids);row['online_seconds']=clock()-tick
        return row
    use_predictor=kind=='cv'
    pred=None
    coefficients=lambda_state.copy() if job.get('lambda_mode')=='ema' else [job.get('coefficient',1.)]*2
    if use_predictor:
        start=clock()
        rr=predictor_features(p,mod,cached,ids) if cached is not None else features(m,tok,mod,cap,ds,ids,micro=job.get('prefix_micro',4))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=job.get('pred_bf16',False)):
            pred=per_example_predictions(p,rr,A,B,s)
        row['prefix_prediction_seconds']=clock()-start
    keep=bool(use_predictor and (job.get('cal_every') or job.get('lambda_mode')=='ema'))
    start=clock();subset=[ids[i] for i in positions]
    labels,truth,_=collect(m,tok,mod,cap,ds,subset,micro=job.get('micro',4),keep=keep,verify_reconstruction=False)
    row['label_seconds']=clock()-start;row['true_gradient_examples']=k
    with torch.no_grad():
        if use_predictor:
            gradients=tuple(((len(ids)/k)*t+c*(p0.sum(0)-(len(ids)/k)*p0[positions].sum(0)))/n for p0,t,c in zip(pred,truth,coefficients))
        else:gradients=tuple(t*(len(ids)/k)/n for t in truth)
    # Coefficient estimates from these labels affect subsequent steps only.
    if job.get('lambda_mode')=='ema':
        with torch.no_grad():
            d=pack(labels);ta,tb=raw(d['x'],d['g'],A,B,s)
            estimates=[]
            for j,(pp,tt) in enumerate(zip(pred,[ta,tb])):
                pp=pp[positions].flatten(1);tt=tt.flatten(1)
                pp=pp-pp.mean(0);tt=tt-tt.mean(0)
                den=float(pp.square().sum());value=float((pp*tt).sum())/max(den,1e-20)
                value=float(np.clip(value,0,job.get('lambda_max',4.)));lambda_state[j]=.9*lambda_state[j]+.1*value;estimates.append(value)
            row['next_coefficient_estimates']=estimates
    oldA=A.detach().clone() if job.get('cal_every') else None
    oldB=B.detach().clone() if job.get('cal_every') else None
    opt.zero_grad(set_to_none=True)
    for w,g in zip([A,B],gradients):
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite gradient')
        w.grad=g.detach()
    row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
    if job.get('cal_every'):
        replay.extend(labels)
        del replay[:-job.get('replay_size',64)]
        if step%job['cal_every']==0:
            row['calibration']=train_calibration(p,calopt,replay,oldA,oldB,s,job,calrng)
            row['calibration_seconds']=row['calibration']['seconds']
    row['coefficients']=coefficients
    row['online_seconds']=clock()-tick
    return row

def run(task,jobs,seed_id,pseed):
    snapshot=EXP/'provenance/source_versions'/CODE_SHA
    snapshot.mkdir(parents=True,exist_ok=True)
    for script in (EXP/'scripts').glob('*.py'):
        if not (snapshot/script.name).exists():shutil.copy2(script,snapshot/script.name)
    warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=seed_id);A,B,s=weights(mod)
    original=[v.detach().clone() for v in [A,B]];basever=versions(m);cap=Capture(mod)
    ds=Dataset(task,'update',tok);dev=Dataset(task,'accuracy_dev',tok);audit=Dataset(task,'gradient_test',tok)
    from fast_backend import configure
    configure(True,8192)
    teacher_cache=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32)
    reserve_worker_cache()
    cached=None;cache_seconds=0.;entries=[]
    for job in jobs:
        from fast_backend import configure
        configure(job.get('fast_loss',False),job.get('token_budget',2048))
        name=f'{task}__{job["name"]}_u{seed_id}_p{pseed}';out=EXP/'models'/name
        if (out/'summary.json').exists():
            entries.extend(json.loads((out/'summary.json').read_text())['entries']);continue
        if deadline(5400):raise RuntimeError('Final validation reserve reached')
        out.mkdir(exist_ok=True);seed(seed_id)
        with torch.no_grad():A.copy_(original[0]);B.copy_(original[1])
        path=Path(job['predictor_path'].format(task=task,pseed=pseed)) if job.get('predictor_path') else predictor_path(task,pseed)
        p=load_predictor(path)[0] if job['kind']=='cv' else None
        if job.get('cal_scope')=='head':
            for parameter_name,parameter in p.named_parameters():parameter.requires_grad_(parameter_name.startswith('head.'))
        opt=torch.optim.AdamW([A,B],lr=job['lr'],weight_decay=0)
        calopt=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=job.get('cal_lr',1e-5),weight_decay=.01) if job.get('cal_every') or job.get('cal_on_refresh') else None
        if job.get('cache') and cached is None:
            cached,cache_seconds=cache_features(m,tok,mod,cap,ds)
            reserve_worker_cache()
            write(EXP/'results'/f'{task}_prefix_cache.json',dict(seconds=cache_seconds,n=len(cached),scope='Frozen module-input X cached; current module output Y recomputed every step',full_inputs=True))
        use_cache=cached if job.get('cache') else None
        rng=np.random.default_rng(seed_id);order=[]
        while len(order)<job['steps']*32:order.extend(map(int,rng.permutation(len(ds))))
        crng=np.random.default_rng(seed_id+10000);calrng=np.random.default_rng(seed_id+20000)
        replay=[];lambda_state=[1.,1.];history=[];ee=[]
        config=dict(task=task,job=job,update_seed=seed_id,predictor_seed=pseed,warmup_sha256=warm['adapter_sha256'],predictor_sha256=sha(path) if p else None,update_sha256=sha(data(task,'update')),full_inputs=True,code_sha256=ENTRY_SHA,source_bundle_sha256=CODE_SHA,calibration='After update, training labels only; no predictor selection with heldout gradients')
        config['teacher_prefix_cache']=dict(seconds=teacher_cache.seconds,bytes=teacher_cache.bytes,examples=len(teacher_cache.rows),full_inputs=True)
        write(out/'config.json',config);started=time.time()
        for st in range(1,job['steps']+1):
            if deadline(3600):raise RuntimeError('Final hour reserved for validation/report')
            ids=order[(st-1)*32:st*32];positions=list(map(int,crng.choice(32,job.get('k',4),replace=False)))
            row=update(m,tok,mod,cap,p,ds,ids,positions,opt,job,st,replay,calopt,calrng,lambda_state,use_cache)
            row.update(indices=ids,positions=positions)
            history.append(row)
            if st in job.get('save',[32,64,128,256]):
                dest=out/f'step-{st}';m.save_pretrained(dest)
                audit_start=clock()
                row['dev_ce']=ce(m,tok,dev,ids=list(range(min(128,len(dev)))),micro=8)
                if p is not None:
                    rr,truth,_=collect(m,tok,mod,cap,audit,list(range(16)),keep=True,verify_reconstruction=False)
                    coefficients=lambda_state if job.get('lambda_mode')=='ema' else [job.get('coefficient',1.)]*2
                    row['gradient_audit']=variance_diagnostic(p,rr,A,B,s,coefficients)
                row['audit_seconds']=clock()-audit_start
                entry=dict(task=task,tag=f'{name}__step{st}',adapter=str(dest),method=job['name'],lr=job['lr'],seed=seed_id,predictor_seed=pseed,steps=st,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=teacher_cache.seconds+(cache_seconds if use_cache is not None else 0),dev_ce=row['dev_ce'],adapter_sha256=sha(dest/'adapter_model.safetensors'))
                ee.append(entry);write(out/'entries.json',ee)
            if st%16==0 or st==1:
                write(out/'history.json',history)
                log(dict(run=name,step=st,online_seconds=sum(r['online_seconds'] for r in history),dev_ce=row.get('dev_ce'),coefficients=row.get('coefficients')))
        if p is not None:save_predictor(p,out/'predictor-final.pt')
        assert versions(m)==basever
        write(out/'history.json',history)
        write(out/'summary.json',dict(complete=True,entries=ee,wall_seconds=time.time()-started,online_seconds=sum(r['online_seconds'] for r in history),calibration_seconds=sum(r['calibration_seconds'] for r in history),true_gradient_examples=sum(r['true_gradient_examples'] for r in history),base_unchanged=True,cache_seconds=teacher_cache.seconds+(cache_seconds if use_cache is not None else 0)))
        entries.extend(ee);write(EXP/'results'/f'{task}_entries_u{seed_id}.json',entries)
        del p,opt,calopt;gc.collect()
    teacher_cache.close();cap.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    run(a.task,json.loads(Path(a.jobs).read_text()),a.seed,a.predictor_seed)

if __name__=='__main__':main()
