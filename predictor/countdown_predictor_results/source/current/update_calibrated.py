"""Matched 16-step online calibration. Held-out targets are diagnostic only."""
from predictor import *
import argparse


def digest_state(state):
    h=hashlib.sha256()
    for k,v in state.items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['calibrated','frozen','oracle'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--train-seed',type=int,required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=16)
    a=ap.parse_args();cfg=json.loads((EXP/'configs/experiment.json').read_text())
    out=EXP/'models'/a.name;out.mkdir(exist_ok=False)
    workspace=EXP.parent.parent
    start=workspace/'predictor_countdown2/models/teacher/step-32'
    source=EXP.parent/'one-step/models'/f'y_mask_pos_s{a.train_seed}'/'best.pt'
    m,tok,mod=setup_model(str(start),rng=a.seed);m.eval()
    A,B,scale=weights(mod);cap=Capture(m,mod,8)
    p,info=load_predictor(source);assert p.source=='y' and p.aux=='mask_pos'
    p.requires_grad_(a.mode=='calibrated');p.eval()
    ds=CompletionDataset(data_path('update'),tok,m.config.max_position_embeddings)
    order_rng=np.random.RandomState(a.seed);subset_rng=np.random.RandomState(a.seed+10000)
    opt=torch.optim.AdamW([A,B],lr=cfg['lora_lr'],weight_decay=0)
    popt=torch.optim.AdamW(p.parameters(),lr=cfg['calibration_lr'],weight_decay=.01) if a.mode=='calibrated' else None
    base_before=digest_state({n:v for n,v in m.named_parameters() if not v.requires_grad})
    p_before=digest_state(p.state_dict());history=[];order=[];start_time=time.time()
    write(out/'config.json',dict(args=vars(a),protocol=cfg,predictor=str(source),adapter=str(start),full_inputs=True))

    def collect(indices):
        samples=[];elapsed=time.time();true_a=torch.zeros_like(A);true_b=torch.zeros_like(B)
        for st in range(0,len(indices),8):
            ids=indices[st:st+8];bb=batch(ds,ids,tok);m.zero_grad(set_to_none=True)
            n=int((bb['labels'][:,1:]!=-100).sum())
            (forward(m,bb).loss*n).backward()
            true_a.add_(A.grad);true_b.add_(B.grad)
            xx=cap.x;yy=cap.y.detach();gg=cap.y.grad.detach()
            # Check analytic A/B reconstruction against true backward in every microbatch.
            pair=raw_grads(xx.flatten(0,1),gg.flatten(0,1),A,B,scale)
            for estimated,target in zip(pair,[A.grad,B.grad]):
                assert measures(estimated,target)['cosine']>.998
            for j,i in enumerate(ids):
                length=len(ds[i]['input_ids']);active=torch.tensor(ds[i]['labels'][1:]+[-100])>=0
                samples.append(dict(index=int(i),x=xx[j,:length].detach().cpu().bfloat16(),
                    y=yy[j,:length].cpu().bfloat16(),g=gg[j,:length].float().cpu(),active=active,n=int(active.sum())))
        m.zero_grad(set_to_none=True)
        return samples,time.time()-elapsed,(true_a,true_b)

    def predict(d):
        return torch.cat([infer(p,d,ii) for ii in torch.arange(len(d['x']),device='cuda').split(8)])

    def diagnose(d,g):
        with torch.no_grad():
            pa,pb=braw(d['x'],g,A.detach(),B.detach(),scale)
            ta,tb=braw(d['x'],d['g'],A.detach(),B.detach(),scale)
            pp=torch.cat([pa.flatten(1),pb.flatten(1)],1)
            tt=torch.cat([ta.flatten(1),tb.flatten(1)],1)
            return dict(activation={k:float(v.mean()) for k,v in metrics(g,d['g']).items()},
                lora={k:float(v.mean()) for k,v in metrics(pp,tt).items()},
                batch_lora=measures(pp.sum(0),tt.sum(0)))

    pre_step_state={k:v.detach().clone() for k,v in p.state_dict().items()} if popt else None
    for step in range(1,a.steps+1):
        t0=time.time()
        if len(order)<32:order=list(order_rng.permutation(len(ds)))
        ix,order=order[:32],order[32:]
        selected=set(subset_rng.choice(32,size=4,replace=False).tolist())
        cal_ids=[i for j,i in enumerate(ix) if j in selected]
        held_ids=[i for j,i in enumerate(ix) if j not in selected]
        # Separate acquisition makes calibration and diagnostics budgets explicit.
        cal_samples,cal_seconds,cal_true=collect(cal_ids)
        dc=pack(cal_samples)
        with torch.no_grad():before_cal=diagnose(dc,predict(dc))
        ab_before=[A.detach().clone(),B.detach().clone()]
        loss_value=None;pnorm=None;fit_start=time.time()
        if popt:
            p.train();g=predict(dc)
            pa,pb=braw(dc['x'],g,A.detach(),B.detach(),scale)
            with torch.no_grad():ta,tb=braw(dc['x'],dc['g'],A.detach(),B.detach(),scale)
            la=(pa-ta).square().sum((1,2))/ta.square().sum((1,2)).clamp_min(1e-12)
            lb=(pb-tb).square().sum((1,2))/tb.square().sum((1,2)).clamp_min(1e-12)
            lg=(g-dc['g']).square().sum()/dc['valid'].sum()/g.shape[-1]/p.target_scale.square()
            loss=(la.mean()+lb.mean())/2+.25*lg
            popt.zero_grad(set_to_none=True);loss.backward()
            assert A.grad is None and B.grad is None
            pnorm=float(nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True));popt.step()
            loss_value=float(loss.detach());p.eval()
        fit_seconds=time.time()-fit_start
        assert torch.equal(A,ab_before[0]) and torch.equal(B,ab_before[1])
        with torch.no_grad():after_cal=diagnose(dc,predict(dc))
        # Targets outside the calibration subset are acquired only after calibration.
        held_samples,held_seconds,held_true=collect(held_ids)
        dh=pack(held_samples)
        with torch.no_grad():after_held=diagnose(dh,predict(dh))
        # Before-calibration held-out metrics use the pre-step predictor snapshot.
        # Keep this measurement independent of any predictor optimization.
        if popt:
            after_state={k:v.detach().clone() for k,v in p.state_dict().items()}
            p.load_state_dict(pre_step_state)
            with torch.no_grad():before_held=diagnose(dh,predict(dh))
            p.load_state_dict(after_state)
        else:before_held=after_held
        by_id={s['index']:s for s in cal_samples+held_samples};full=pack([by_id[int(i)] for i in ix])
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            pred=full['g'] if a.mode=='oracle' else predict(full)
            aggregate=diagnose(full,pred)
            ga=torch.zeros_like(A);gb=torch.zeros_like(B)
            for ids in torch.arange(32,device='cuda').split(8):
                aa,bb=raw_grads(full['x'][ids].flatten(0,1),pred[ids].flatten(0,1),A,B,scale)
                ga+=aa;gb+=bb
            denom=int(full['counts'].sum())
            if a.mode=='oracle':ga=cal_true[0]+held_true[0];gb=cal_true[1]+held_true[1]
            A.grad=ga/denom;B.grad=gb/denom
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step()
        row=dict(step=step,sample_indices=[int(i) for i in ix],calibration_indices=[int(i) for i in cal_ids],
            heldout_indices=[int(i) for i in held_ids],supervised_tokens=denom,calibration_loss=loss_value,
            calibration_gradient_norm=pnorm,lora_gradient_norm=norm,calibration_before=before_cal,calibration_after=after_cal,
            heldout_before=before_held,heldout_after=after_held,applied_gradient=aggregate,
            timing=dict(calibration_targets=cal_seconds,diagnostic_targets=held_seconds,calibration_fit=fit_seconds,total=time.time()-t0),
            true_target_examples=dict(calibration=4 if popt else 0,oracle=32 if a.mode=='oracle' else 0,
                diagnostics=28 if popt else (0 if a.mode=='oracle' else 32)))
        history.append(row);write(out/'history.json',history);print(json.dumps(row),flush=True)
        if step in cfg['evaluation_steps'] or step==a.steps:
            m.save_pretrained(out/f'step-{step}')
            torch.save(dict(config=info['config'],state={k:v.detach().cpu() for k,v in p.state_dict().items()},step=step,
                args=vars(a)),out/f'predictor-step-{step}.pt')
        if popt:pre_step_state={k:v.detach().clone() for k,v in p.state_dict().items()}
    base_after=digest_state({n:v for n,v in m.named_parameters() if not v.requires_grad})
    assert base_before==base_after
    p_after=digest_state(p.state_dict());assert (p_before!=p_after)==(a.mode=='calibrated')
    torch.save(dict(lora_optimizer=opt.state_dict(),predictor_optimizer=popt.state_dict() if popt else None),out/'optimizer-final.pt')
    cap.close();write(out/'audit.json',dict(complete=True,steps=a.steps,base_unchanged=True,predictor_changed=p_before!=p_after,
        calibration_excludes_heldout=True,lora_unchanged_during_calibration=True,predictor_initial_hash=p_before,predictor_final_hash=p_after,elapsed=time.time()-start_time))

if __name__=='__main__':main()
