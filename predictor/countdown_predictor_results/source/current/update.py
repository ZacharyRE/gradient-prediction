"""Direct gradient updates of one LoRA. Frozen hidden-only predictor; prefix forward only."""
from predictor import *
import argparse
class PrefixDone(Exception):pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--adapter',required=True)
    ap.add_argument('--predictor');ap.add_argument('--mode',choices=['synthetic','true','mean','dense_mean','zero'],default='synthetic')
    ap.add_argument('--mean-file');ap.add_argument('--steps',type=int,default=192);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--save',nargs='+',type=int,default=[1,4,16,48,96,192]);ap.add_argument('--optimizer',choices=['adamw','sgd'],default='sgd')
    ap.add_argument('--weight-decay',type=float,default=0.);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--seed',type=int,default=101)
    ap.add_argument('--data',default='update');a=ap.parse_args();out=EXP/'models'/a.name;out.mkdir(exist_ok=False)
    m,tok,mod=setup_model(a.adapter,rng=a.seed);m.eval();A,B,scale=weights(mod);cap=Capture(m,mod,8)
    ds=CompletionDataset(data_path(a.data),tok,m.config.max_position_embeddings)
    p=None;handle=None
    if a.mode=='synthetic':
        p,info=load_predictor(a.predictor);p.requires_grad_(False);p.eval()
        def stop(module,args,output):raise PrefixDone()
        handle=mod.register_forward_hook(stop)
    if a.mode in ['mean','dense_mean']:mean=torch.load(a.mean_file,weights_only=True).cuda()
    opt=(torch.optim.SGD if a.optimizer=='sgd' else torch.optim.AdamW)([A,B],lr=a.lr,weight_decay=a.weight_decay)
    frozen={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in m.named_parameters() if not v.requires_grad}
    p_before={k:v.clone() for k,v in p.state_dict().items()} if p is not None else {}
    initialA=A.detach().clone();initialB=B.detach().clone();hist=[];order=[];start=time.time()
    write(out/'config.json',dict(args=vars(a),base=MODEL,module='model.layers.8.self_attn.o_proj',
        true_backward_for_update=a.mode=='true',prefix_forward_only=a.mode=='synthetic',full_inputs=True,predictor_frozen=True))
    for step in range(1,a.steps+1):
        if len(order)<a.batch_size:order=list(np.random.permutation(len(ds)))
        ix,order=order[:a.batch_size],order[a.batch_size:];opt.zero_grad(set_to_none=True)
        denom=sum(sum(t!=-100 for t in ds[i]['labels'][1:]) for i in ix);total=0.
        if a.mode=='mean':A.grad=mean[:A.numel()].reshape_as(A).clone();B.grad=mean[A.numel():].reshape_as(B).clone()
        elif a.mode=='dense_mean':A.grad=scale*(B.detach().T@mean);B.grad=scale*(mean@A.detach().T)
        elif a.mode=='zero':A.grad=torch.zeros_like(A);B.grad=torch.zeros_like(B)
        else:
            for st in range(0,len(ix),8):
                ids=ix[st:st+8];bb=batch(ds,ids,tok)
                if a.mode=='true':
                    n=int((bb['labels'][:,1:]!=-100).sum());loss=forward(m,bb).loss;(loss*n/denom).backward();total+=float(loss)*n
                else:
                    with torch.no_grad():
                        try:forward(m,{k:v for k,v in bb.items() if k!='labels'})
                        except PrefixDone:pass
                        valid=bb['attention_mask'].bool();active=torch.zeros_like(valid);active[:,:-1]=bb['labels'][:,1:]>=0
                        pos=torch.arange(valid.shape[1],device='cuda')[None,:]/(valid.sum(1,keepdim=True)-1).clamp_min(1)
                        secondary=cap.residual+cap.y.detach() if p.source=='h' else cap.y.detach()
                        pred=p(cap.x,secondary,active,pos,valid)
                        ga,gb=raw_grads(cap.x.flatten(0,1),pred.flatten(0,1),A,B,scale)
                        if A.grad is None:A.grad=ga/denom;B.grad=gb/denom
                        else:A.grad.add_(ga,alpha=1/denom);B.grad.add_(gb,alpha=1/denom)
        norm=torch.nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True);opt.step()
        row=dict(step=step,loss=total/denom if a.mode=='true' else None,gradient_norm=float(norm),
            sample_indices=[int(i) for i in ix],supervised_tokens=int(denom),
            A_delta=float((A-initialA).norm()),B_delta=float((B-initialB).norm()),seconds=time.time()-start)
        hist.append(row)
        if step%16==0 or step==a.steps:print(json.dumps(row),flush=True);write(out/'history.json',hist)
        if step in a.save:m.save_pretrained(out/f'step-{step}')
    if handle:handle.remove()
    cap.close()
    after={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in m.named_parameters() if not v.requires_grad}
    assert frozen==after
    if p is not None:assert all(torch.equal(v,p.state_dict()[k]) for k,v in p_before.items())
    write(out/'audit.json',dict(base_unchanged=True,predictor_unchanged=True,trainable_names=[n for n,v in m.named_parameters() if v.requires_grad],
        true_backward_calls=0 if a.mode!='true' else a.steps*((a.batch_size+7)//8),elapsed=time.time()-start))
if __name__=='__main__':main()
