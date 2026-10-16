"""Paid local suffix linearization, with matched exact-objective controls."""
from reuse_train import *
from tangent_suffix import TangentSuffix,exact_class_gradients
from safetensors.torch import load_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warmup-seed',type=int,default=301);ap.add_argument('--subset',default='candidate_0');ap.add_argument('--seed',type=int,default=401);ap.add_argument('--jobs',required=True);ap.add_argument('--suffix',default='_tangent');ap.add_argument('--anchor-adapter');ap.add_argument('--skip-initial-audit',action='store_true');a=ap.parse_args()
    jobs=json.loads(Path(a.jobs).read_text());warm=json.loads((EXP/'configs'/f'mnli_warmup_w{a.warmup_seed}.json').read_text());started=time.time()
    m,tok,mod=setup('1p5b',warm['adapter'],rng=a.seed);A,B,s=weights(mod);initial=(A.detach().clone(),B.detach().clone());basever=versions(m);cap=Capture(mod);ds=Dataset('mnli',a.subset,tok);configure(True,8192);cache=TeacherPrefixCache(m,tok,mod,cap,ds,micro=32)
    if a.anchor_adapter:
        anchor_values=load_file(str(Path(a.anchor_adapter)/'adapter_model.safetensors'))
        with torch.no_grad():
            A.copy_(next(v for k,v in anchor_values.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in anchor_values.items() if k.endswith('lora_B.weight')))
    needs_tangent=any(j['kind'] in ['linear','linear_cv_full','linear_cv_class','hybrid_full','hybrid_class'] for j in jobs)
    tangent0=TangentSuffix(m,tok,mod,cap,ds,cache,micro=16) if needs_tangent else None;class_tokens=torch.tensor([tok.encode(c,add_special_tokens=False)[0] for c in 'ABC'],device='cuda')
    if tangent0 is not None and not a.skip_initial_audit:
        pred,_=tangent0.predict([0,1,2,3],mod);truth,_=exact_class_gradients(m,tok,mod,cap,ds,[0,1,2,3],class_tokens,micro=4);audit=measures(flat([v.sum(0) for v in pred]),flat(truth));assert audit['cosine']>.99 and audit['relative_l2']<.05,audit
        audit_path=EXP/'results'/f'tangent_initial_audit_w{a.warmup_seed}_{a.subset}_u{a.seed}{a.suffix}.json';write(audit_path,dict(passed=True,anchor_adapter=a.anchor_adapter or warm['adapter'],initial_exact_class_gradient_comparison=audit,jacobian_dtype='bfloat16',jacobian_collection_seconds=tangent0.seconds,jacobian_bytes=tangent0.bytes))
    with torch.no_grad():A.copy_(initial[0]);B.copy_(initial[1])
    reserve_worker_cache();entries=[]
    for job in jobs:
        name=f'mnli__w{a.warmup_seed}_{a.subset}_{job["name"]}_u{a.seed}{a.suffix}';out=EXP/'models'/name
        if (out/'summary.json').exists():
            previous=json.loads((out/'config.json').read_text());assert previous['job']==job and previous.get('anchor_adapter')==(a.anchor_adapter or warm['adapter']),'Cannot resume a changed tangent recipe'
            entries.extend(json.loads((out/'summary.json').read_text())['entries']);continue
        if deadline(5400):raise RuntimeError('Final validation reserve')
        out.mkdir(exist_ok=True);seed(a.seed)
        with torch.no_grad():A.copy_(initial[0]);B.copy_(initial[1])
        tangent=tangent0;opt=torch.optim.AdamW([A,B],lr=job['lr'],weight_decay=0);rng=np.random.default_rng(a.seed);crng=np.random.default_rng(a.seed+10000);order=[]
        while len(order)<job['steps']*32:order.extend(map(int,rng.permutation(len(ds))))
        use_tangent=job['kind'] in ['linear','linear_cv_full','linear_cv_class','hybrid_full','hybrid_class'];prep=cache.seconds+(tangent0.seconds if use_tangent else 0.);history=[];ee=[];held_gradients=None;optimizer_steps=0
        objective='full_answer' if job['kind'] in ['linear_cv_full','oracle_full','hold_full','skip_full'] else 'hybrid_full_true_class_surrogate' if job['kind']=='hybrid_full' else 'restricted_class'
        write(out/'config.json',dict(job=job,warmup_seed=a.warmup_seed,update_seed=a.seed,subset=a.subset,warmup_sha256=warm['adapter_sha256'],anchor_adapter=a.anchor_adapter or warm['adapter'],anchor_sha256=sha(Path(a.anchor_adapter or warm['adapter'])/'adapter_model.safetensors'),data_sha256=sha(data('mnli',a.subset)),source_sha256=source_signature(__file__),teacher_cache_seconds=cache.seconds,initial_jacobian_seconds=tangent0.seconds if use_tangent else 0.,initial_jacobian_backward_examples=tangent0.backwards if use_tangent else 0,prep_seconds=prep,full_inputs=True,loss_objective=objective,linearization='Teacher ABC logit differences linearized in LoRA output activation; current LoRA B A product retained'))
        for st in range(1,job['steps']+1):
            if deadline(900):raise RuntimeError('Compute deadline')
            tick=clock();refresh_seconds=0.;refresh_backwards=0
            if use_tangent and job.get('refresh_every') and st>1 and (st-1)%job['refresh_every']==0:
                tangent=TangentSuffix(m,tok,mod,cap,ds,cache,micro=16);refresh_seconds=tangent.seconds;refresh_backwards=tangent.backwards
            ids=order[(st-1)*32:st*32];periodic=job['kind'] in ['hybrid_full','hybrid_class','hold_full','skip_full'];true_step=periodic and (st-1)%job.get('true_every',4)==0
            k=(32 if true_step else 0) if periodic else job.get('k',0);positions=list(range(32)) if k==32 else list(map(int,crng.choice(32,k,replace=False))) if k else []
            prediction_seconds=0.;label_seconds=0.;loss=None
            if use_tangent and not true_step:
                t=clock();pred,loss=tangent.predict(ids,mod);prediction_seconds=clock()-t
            if periodic:
                n=32 if job['kind']=='hybrid_class' else sum(ds[i]['n'] for i in ids)
                if true_step:
                    t=clock()
                    if job['kind']=='hybrid_class':truth,loss=exact_class_gradients(m,tok,mod,cap,ds,ids,class_tokens,micro=32);loss/=32
                    else:_,truth,_=collect(m,tok,mod,cap,ds,ids,micro=32,keep=False,verify_reconstruction=False)
                    label_seconds=clock()-t;gradients=tuple(g/n for g in truth)
                    if job['kind']=='hold_full':held_gradients=tuple(g.detach().clone() for g in gradients)
                elif use_tangent:gradients=tuple(p.sum(0)/n for p in pred)
                elif job['kind']=='hold_full':gradients=tuple(g.clone() for g in held_gradients)
                else:gradients=None
            elif job['kind']=='linear':gradients=tuple(p.sum(0)/32 for p in pred)
            else:
                t=clock();subset=[ids[i] for i in positions]
                if job['kind'] in ['linear_cv_full','oracle_full']:_,truth,_=collect(m,tok,mod,cap,ds,subset,micro=32,keep=False,verify_reconstruction=False);n=sum(ds[i]['n'] for i in ids)
                else:truth,loss=exact_class_gradients(m,tok,mod,cap,ds,subset,class_tokens,micro=32);loss/=k;n=32
                label_seconds=clock()-t
                if job['kind'] in ['oracle_class','oracle_full']:gradients=tuple(t*(32/k)/n for t in truth)
                else:gradients=tuple((p.sum(0)+(32/k)*(t-p[positions].sum(0)))/n for p,t in zip(pred,truth))
            opt.zero_grad(set_to_none=True)
            norm=None
            if gradients is not None:
                for w,g in zip([A,B],gradients):
                    assert torch.isfinite(g).all();w.grad=g.detach()
                norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();optimizer_steps+=1
            if gradients is not None and job.get('anchor_decay',0):
                # Decoupled attraction to the starting adapter; matched controls
                # must use the identical rule. This is an objective/recipe change.
                with torch.no_grad():
                    for w,w0 in zip([A,B],initial):w.add_(w0-w,alpha=job['lr']*job['anchor_decay'])
            m.zero_grad(set_to_none=True)
            row=dict(step=st,optimizer_steps=optimizer_steps,online_seconds=clock()-tick,true_gradient_examples=k,gradient_norm=norm,loss=loss,prediction_seconds=prediction_seconds,label_seconds=label_seconds,jacobian_refresh_seconds=refresh_seconds,jacobian_refresh_backward_examples=refresh_backwards,indices=ids,positions=positions);history.append(row)
            if st in job.get('save',[32,128,256,512]):
                dest=out/f'step-{st}';t=clock();m.save_pretrained(dest);save_seconds=clock()-t
                entry=dict(task='mnli',tag=f'{name}__step{st}',adapter=str(dest),method=job['name'],kind=job['kind'],k=job.get('k',0),true_every=job.get('true_every'),optimizer_steps=optimizer_steps,lr=job['lr'],seed=a.seed,warmup_seed=a.warmup_seed,subset=a.subset,predictor_seed=None,steps=st,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=prep,adapter_sha256=sha(dest/'adapter_model.safetensors'),save_seconds=save_seconds,diagnostic_seconds=0.,loss_objective=objective);ee.append(entry);write(out/'entries.json',ee);log(entry)
            if st%32==0:write(out/'history.json',history)
        assert versions(m)==basever
        write(out/'summary.json',dict(complete=True,entries=ee,online_seconds=sum(r['online_seconds'] for r in history),cache_seconds=prep,true_gradient_examples=sum(r['true_gradient_examples'] for r in history),initial_jacobian_backward_examples=tangent0.backwards if use_tangent else 0,jacobian_refresh_backward_examples=sum(r['jacobian_refresh_backward_examples'] for r in history),base_unchanged=True));entries.extend(ee);del opt,tangent;gc.collect()
    cache.close();cap.close();write(EXP/'results'/f'entries_mnli_w{a.warmup_seed}_{a.subset}_u{a.seed}{a.suffix}.json',entries);log(dict(complete=True,entries=len(entries),wall_seconds=time.time()-started))

if __name__=='__main__':main()
