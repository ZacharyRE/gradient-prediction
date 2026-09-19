"""Refresh a hidden-only predictor; optional hybrid adds a true-probe mean to neural residuals."""
from predictor import *
import argparse, copy


class PrefixDone(Exception):
    pass


def collect_current(m, tok, cap, ds, ids):
    samples = []
    for start in range(0, len(ids), 8):
        ix = ids[start:start+8]
        bb = batch(ds, ix, tok)
        m.zero_grad(set_to_none=True)
        n = int((bb['labels'][:, 1:] != -100).sum())
        (forward(m, bb).loss * n).backward()
        for j, i in enumerate(ix):
            length = len(ds[i]['input_ids'])
            active = torch.tensor(ds[i]['labels'][1:] + [-100]) >= 0
            samples.append(dict(index=int(i), x=cap.x[j, :length].detach().cpu().bfloat16(),
                y=cap.y[j, :length].detach().cpu().bfloat16(), g=cap.y.grad[j, :length].detach().cpu().float(),
                active=active, n=int(active.sum())))
    m.zero_grad(set_to_none=True)
    return samples


@torch.no_grad()
def evaluate_batch(p, d, meta):
    r, pp, tt, _ = evaluate(p, d, meta, True)
    counts = d['counts'].cpu()
    bp, bt = [], []
    for st in range(0, len(pp), 32):
        c = counts[st:st+32, None]
        bp.append((pp[st:st+32]*c).sum(0)/c.sum())
        bt.append((tt[st:st+32]*c).sum(0)/c.sum())
    batch_metrics = metrics(torch.stack(bp), torch.stack(bt))
    r['batch32_lora'] = {k: dict(mean=float(v.mean()), median=float(v.median()),
        std=float(v.std()) if len(v)>1 else None, n_batches=len(v)) for k,v in batch_metrics.items()}
    r['refresh_selection_score'] = r['selection_score'] + r['batch32_lora']['relative_l2']['mean']**2
    return r


def calibrate(p, train, dev, meta, epochs=30, lr=1e-4, adam_state=None, adam_step=1, adam_weight=0.):
    d, vd = pack(train), pack(dev)
    A, B, scale = meta['A'].cuda(), meta['B'].cuda(), meta['scale']
    with torch.no_grad():
        ta, tb = braw(d['x'], d['g'], A, B, scale)
        energy = d['g'].square().sum()/d['valid'].sum()/896
    def virtual_delta(gradients):
        vec=torch.cat([v.flatten() for v in gradients])
        factor=(1./(vec.norm()+1e-6)).clamp(max=1.)
        chunks=[]
        for i,g in enumerate(gradients):
            g=g*factor;mom,var=adam_state[i]
            mm=.9*mom+.1*g;vv=.999*var+.001*g.square()
            chunks.append((mm/(1-.9**adam_step))/(vv.sqrt()/(1-.999**adam_step)**.5+1e-8))
        return torch.cat([v.flatten() for v in chunks])
    @torch.no_grad()
    def assess():
        r=evaluate_batch(p,vd,meta)
        if adam_weight:
            errors=[]
            for ix in torch.arange(len(dev),device='cuda').split(32):
                pred=infer(p,vd,ix)
                pg=braw(vd['x'][ix],pred,A,B,scale);tg=braw(vd['x'][ix],vd['g'][ix],A,B,scale)
                n=vd['counts'][ix].sum()
                pd=virtual_delta([v.sum(0)/n for v in pg]);td=virtual_delta([v.sum(0)/n for v in tg])
                errors.append((pd-td).square().sum()/td.square().sum().clamp_min(1e-12))
            value=float(torch.stack(errors).mean());r['adam_delta_relative_mse']=value
            r['refresh_selection_score']+=adam_weight*value
        return r
    before = assess()
    best_score = before['refresh_selection_score']
    best = {k: v.detach().clone() for k, v in p.state_dict().items()}
    best_epoch = 0
    p.requires_grad_(True)
    opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=.01)
    history = []
    for epoch in range(1, epochs+1):
        p.train()
        for ix in torch.randperm(len(train), device='cuda').split(32):
            pred = infer(p, d, ix)
            pa, pb = braw(d['x'][ix], pred, A, B, scale)
            individual, aggregate = [], []
            for v, t in [(pa, ta[ix]), (pb, tb[ix])]:
                individual.append(((v-t).square().sum((1,2))/t.square().sum((1,2)).clamp_min(1e-12)).mean())
                aggregate.append((v.sum(0)-t.sum(0)).square().sum()/t.sum(0).square().sum().clamp_min(1e-12))
            token = (pred-d['g'][ix]).square().sum()/d['valid'][ix].sum()/896/energy
            loss = sum(individual)/2 + sum(aggregate)/2 + .25*token
            if adam_weight:
                n=d['counts'][ix].sum()
                pd=virtual_delta([pa.sum(0)/n,pb.sum(0)/n])
                td=virtual_delta([ta[ix].sum(0)/n,tb[ix].sum(0)/n]).detach()
                loss=loss+adam_weight*(pd-td).square().sum()/td.square().sum().clamp_min(1e-12)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        if epoch % 5 == 0 or epoch == epochs:
            result = assess()
            score = result['refresh_selection_score']
            history.append(dict(epoch=epoch, score=score))
            if score < best_score:
                best_score, best_epoch = score, epoch
                best = {k: v.detach().clone() for k, v in p.state_dict().items()}
    p.load_state_dict(best)
    p.requires_grad_(False)
    p.eval()
    after = assess()
    return dict(before=before, after=after, best_epoch=best_epoch, history=history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=104)
    ap.add_argument('--predictor', default=str(EXP/'models/ablation_y_none/best.pt'))
    ap.add_argument('--refresh', nargs='+', type=int, default=[2,5,9,13])
    ap.add_argument('--probe-seed', type=int, default=701)
    ap.add_argument('--probe-count', type=int, default=128)
    ap.add_argument('--dev-count', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--predictor-lr', type=float, default=1e-4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--adam-loss-weight', type=float, default=0.)
    ap.add_argument('--mean-residual-weight', type=float, default=None)
    a = ap.parse_args()
    assert all(1 <= x <= 16 for x in a.refresh)
    assert a.mean_residual_weight is None or 0 <= a.mean_residual_weight <= 1
    out = EXP/'models'/a.name
    out.mkdir(exist_ok=False)
    m, tok, mod = setup_model(EXP/'models/teacher/step-32', rng=a.seed)
    A, B, scale = weights(mod)
    cap = Capture(m, mod, 8)
    m.eval()
    p, info = load_predictor(a.predictor)
    p.requires_grad_(False)
    base_hash = {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in m.named_parameters() if not v.requires_grad}
    ds = CompletionDataset(data_path('update'), tok, m.config.max_position_embeddings)
    probes = CompletionDataset(data_path('predictor_train'), tok, m.config.max_position_embeddings)
    dev = CompletionDataset(data_path('predictor_dev'), tok, m.config.max_position_embeddings)
    # Preserve the baseline update order; calibration uses only torch randomness.
    update_order = np.random.permutation(len(ds)).tolist()
    probe_ids = np.random.default_rng(a.probe_seed).permutation(len(probes))[:a.probe_count].tolist()
    dev_ids = list(range(a.dev_count))
    write(out/'config.json', dict(args=vars(a), update_order=update_order[:512], probe_ids=probe_ids,
        dev_ids=dev_ids, predictor_inputs=('current hidden only; true gradients used only as calibration labels' if a.mean_residual_weight is None else 'current hidden only for neural head; true probe mean also contributes directly to LoRA estimator'),
        predictor_supervision='separate predictor_train probes and predictor_dev checkpoint selection',
        max_steps=16, full_inputs=True, no_true_gradient_direct_lora_updates=a.mean_residual_weight is None,
        update_estimator='neural_G_only' if a.mean_residual_weight is None else 'true_probe_mean_plus_centered_neural_residual'))
    opt = torch.optim.AdamW([A,B], lr=a.lr, weight_decay=0.)
    rows, refreshes = [], []
    mean_true_dense=mean_pred_dense=None
    begin = time.time()
    for step in range(1,17):
        if step in a.refresh:
            before_factors = [A.detach().clone(), B.detach().clone()]
            train = collect_current(m,tok,cap,probes,probe_ids)
            validation = collect_current(m,tok,cap,dev,dev_ids)
            meta = dict(A=A.detach().cpu().clone(), B=B.detach().cpu().clone(), scale=scale)
            torch.save(dict(train=train,dev=validation,meta=meta), out/f'calibration-before-{step}.pt')
            adam_state=[(opt.state.get(v,{}).get('exp_avg',torch.zeros_like(v)).detach().clone(),
                         opt.state.get(v,{}).get('exp_avg_sq',torch.zeros_like(v)).detach().clone()) for v in [A,B]]
            result = calibrate(p, train, validation, meta, a.epochs, a.predictor_lr,
                               adam_state=adam_state,adam_step=step,adam_weight=a.adam_loss_weight)
            if a.mean_residual_weight is not None:
                with torch.no_grad():
                    packed=pack(train)
                    estimates=torch.cat([infer(p,packed,ix) for ix in torch.arange(len(train),device='cuda').split(32)])
                    xx=packed['x'].flatten(0,1).float();denom_probe=packed['counts'].sum()
                    mean_true_dense=packed['g'].flatten(0,1).T@xx/denom_probe
                    mean_pred_dense=estimates.flatten(0,1).T@xx/denom_probe
                    torch.save(dict(true_dense=mean_true_dense.cpu(),pred_dense=mean_pred_dense.cpu(),weight=a.mean_residual_weight,
                        probe_ids=probe_ids,supervised_tokens=int(denom_probe)),out/f'hybrid-before-{step}.pt')
                    del packed,estimates,xx
            assert torch.equal(A,before_factors[0]) and torch.equal(B,before_factors[1])
            refreshes.append(dict(before_step=step, **result))
            write(out/'refreshes.json', refreshes)
            print(json.dumps(dict(name=a.name,refresh=step,epoch=result['best_epoch'],
                before=result['before']['batch32_lora'],after=result['after']['batch32_lora'])),flush=True)
        ix = update_order[(step-1)*32:step*32]
        opt.zero_grad(set_to_none=True)
        denom = sum(sum(t != -100 for t in ds[i]['labels'][1:]) for i in ix)
        def stop(module,args,output):
            raise PrefixDone()
        handle = mod.register_forward_hook(stop)
        try:
            for st in range(0,32,8):
                bb = batch(ds,ix[st:st+8],tok)
                with torch.no_grad():
                    try: forward(m,{k:v for k,v in bb.items() if k!='labels'})
                    except PrefixDone: pass
                    valid=bb['attention_mask'].bool()
                    active=torch.zeros_like(valid);active[:,:-1]=bb['labels'][:,1:]>=0
                    pos=torch.arange(valid.shape[1],device='cuda')[None,:]/(valid.sum(1,keepdim=True)-1).clamp_min(1)
                    pred=p(cap.x,cap.y.detach(),active,pos,valid)
                    ga,gb=raw_grads(cap.x.flatten(0,1),pred.flatten(0,1),A,B,scale)
                    if A.grad is None: A.grad=ga/denom;B.grad=gb/denom
                    else: A.grad.add_(ga,alpha=1/denom);B.grad.add_(gb,alpha=1/denom)
        finally: handle.remove()
        if mean_true_dense is not None:
            with torch.no_grad():
                centered=mean_true_dense-a.mean_residual_weight*mean_pred_dense
                A.grad.mul_(a.mean_residual_weight).add_(scale*(B.detach().T@centered))
                B.grad.mul_(a.mean_residual_weight).add_(scale*(centered@A.detach().T))
        norm=torch.nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True)
        opt.step()
        rows.append(dict(step=step,gradient_norm=float(norm),seconds=time.time()-begin))
        # Preserve every state and optimizer moment for post-hoc audits of the applied updates.
        m.save_pretrained(out/f'step-{step}')
        torch.save(dict(config=info['config'],state={k:v.detach().cpu() for k,v in p.state_dict().items()},
            args=vars(a),step=step),out/f'predictor-step-{step}.pt')
        torch.save(opt.state_dict(),out/f'optimizer-step-{step}.pt')
        write(out/'history.json',rows)
    after_hash={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in m.named_parameters() if not v.requires_grad}
    assert after_hash==base_hash
    cap.close()
    write(out/'audit.json',dict(base_unchanged=True,calibration_never_updates_lora=True,
        true_backward_direct_update_calls=0,oracle_calibration_examples=len(a.refresh)*(a.probe_count+a.dev_count),
        oracle_training_examples=len(a.refresh)*a.probe_count,oracle_validation_examples=len(a.refresh)*a.dev_count,
        oracle_backward_calls=len(a.refresh)*((a.probe_count+7)//8+(a.dev_count+7)//8),
        synthetic_update_examples=512,predictor_refreshes=len(a.refresh),elapsed=time.time()-begin,
        uses_calibration_mean_directly_in_lora_gradient=a.mean_residual_weight is not None))
    print('Completed '+a.name,flush=True)


if __name__=='__main__':main()
