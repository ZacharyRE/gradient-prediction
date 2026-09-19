"""Matched online dense-mean baseline: same first prediction, refreshed probe means later."""
from dynamic16_online import collect_current, PrefixDone
from predictor import *
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--refresh',nargs='+',type=int,default=[2,5,9,13]);ap.add_argument('--probe-count',type=int,default=128)
    ap.add_argument('--probe-seed',type=int,default=701)
    ap.add_argument('--dev-count',type=int,default=64);a=ap.parse_args()
    out=EXP/'models'/a.name;out.mkdir(exist_ok=False)
    m,tok,mod=setup_model(EXP/'models/teacher/step-32',rng=a.seed);m.eval();A,B,scale=weights(mod);cap=Capture(m,mod,8)
    frozen={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in m.named_parameters() if not v.requires_grad}
    p,info=load_predictor(EXP/'models/ablation_y_none/best.pt');p.requires_grad_(False)
    ds=CompletionDataset(data_path('update'),tok,m.config.max_position_embeddings)
    probes=CompletionDataset(data_path('predictor_train'),tok,m.config.max_position_embeddings)
    dev=CompletionDataset(data_path('predictor_dev'),tok,m.config.max_position_embeddings)
    order=np.random.permutation(len(ds)).tolist()
    ids=np.random.default_rng(a.probe_seed).permutation(len(probes))[:a.probe_count].tolist()
    opt=torch.optim.AdamW([A,B],lr=.0003,weight_decay=0.)
    write(out/'config.json',dict(args=vars(a),first_step='same original predictor as online neural methods',
        subsequent_steps='current-factor projection of refreshed mean dense gradient; no current-example conditioning',
        probe_ids=ids,update_order=order[:512],validation_role='diagnostic only, not included in mean',full_inputs=True))
    D=None;hist=[];begin=time.time()
    for step in range(1,17):
        if step in a.refresh:
            old=[A.detach().clone(),B.detach().clone()]
            samples=collect_current(m,tok,cap,probes,ids)
            validation=collect_current(m,tok,cap,dev,list(range(a.dev_count)))
            with torch.no_grad():
                D=torch.zeros(896,896,device='cuda');n=0
                for s in samples:
                    D.add_(s['g'].cuda().float().T@s['x'].cuda().float());n+=s['n']
                D.div_(n)
                v=pack(validation);ta,tb=braw(v['x'],v['g'],A.detach(),B.detach(),scale)
                true=torch.cat([ta.flatten(1),tb.flatten(1)],1)/v['counts'][:,None]
                mean=flat([scale*B.detach().T@D,scale*D@A.detach().T])
                stats=summarize(metrics(mean[None].expand_as(true),true))
            assert torch.equal(A,old[0]) and torch.equal(B,old[1])
            torch.save(dict(D=D.cpu(),meta=dict(A=A.detach().cpu(),B=B.detach().cpu(),scale=scale)),out/f'mean-before-{step}.pt')
            write(out/f'mean-before-{step}.json',dict(validation_lora=stats,training_tokens=n))
            del samples,validation,v,ta,tb,true
        opt.zero_grad(set_to_none=True)
        if D is not None:
            with torch.no_grad():A.grad=scale*(B.detach().T@D);B.grad=scale*(D@A.detach().T)
        else:
            ix=order[(step-1)*32:step*32];denom=sum(sum(t!=-100 for t in ds[i]['labels'][1:]) for i in ix)
            def stop(module,args,output):raise PrefixDone()
            handle=mod.register_forward_hook(stop)
            try:
                for st in range(0,32,8):
                    bb=batch(ds,ix[st:st+8],tok)
                    with torch.no_grad():
                        try:forward(m,{k:v for k,v in bb.items() if k!='labels'})
                        except PrefixDone:pass
                        valid=bb['attention_mask'].bool();active=torch.zeros_like(valid);active[:,:-1]=bb['labels'][:,1:]>=0
                        pos=torch.arange(valid.shape[1],device='cuda')[None,:]/(valid.sum(1,keepdim=True)-1).clamp_min(1)
                        pred=p(cap.x,cap.y.detach(),active,pos,valid)
                        ga,gb=raw_grads(cap.x.flatten(0,1),pred.flatten(0,1),A,B,scale)
                        if A.grad is None:A.grad=ga/denom;B.grad=gb/denom
                        else:A.grad.add_(ga,alpha=1/denom);B.grad.add_(gb,alpha=1/denom)
            finally:handle.remove()
        norm=torch.nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True);opt.step()
        hist.append(dict(step=step,gradient_norm=float(norm),seconds=time.time()-begin))
        if step in [1,2,4,8,16]:m.save_pretrained(out/f'step-{step}')
        write(out/'history.json',hist)
    after={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in m.named_parameters() if not v.requires_grad}
    assert frozen==after;cap.close()
    write(out/'audit.json',dict(base_unchanged=True,oracle_calibration_examples=len(a.refresh)*(a.probe_count+a.dev_count),
        oracle_training_examples=len(a.refresh)*a.probe_count,oracle_validation_examples=len(a.refresh)*a.dev_count,
        direct_update_batch_backward_calls=0,refreshes=len(a.refresh),max_steps=16,elapsed=time.time()-begin))
    print('Completed '+a.name,flush=True)
if __name__=='__main__':main()
