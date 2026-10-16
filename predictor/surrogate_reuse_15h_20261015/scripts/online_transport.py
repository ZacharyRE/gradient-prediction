"""Online class surrogates with explicit LoRA-factor transport.

For one example C=G.T@X, so grad_A=s*B.T@C and grad_B=s*C@A.T.
Store a delayed class-conditional estimate of C instead of stale parameter
gradients. The correction subset remains uniformly sampled and unbiased before
Adam/clipping. The matched flat-gradient class EMA isolates transport's value.
"""
from predictor_core import *
from gpu_features import collect_gpu
from updates_legacy import clock

class ClassSurrogate:
    def __init__(self,mod,source_rows,kind,forget=.99,ridge=10.):
        A,B,s=weights(mod);self.kind=kind;self.forget=forget;self.scale=s
        classes={v:i for i,v in enumerate(sorted({r['gold'] for r in source_rows}))};self.label_list=[classes[r['gold']] for r in source_rows];self.labels=torch.tensor(self.label_list,device='cuda')
        shape=(B.shape[0],A.shape[1]) if kind=='transport' else (A.numel()+B.numel(),)
        self.total=torch.zeros(len(classes),*shape,device='cuda');self.mass=torch.full((len(classes),),float(ridge),device='cuda',dtype=torch.float32)

    @torch.no_grad()
    def predict(self,A,B):
        avg=self.total/self.mass.reshape((-1,)+(1,)*(self.total.ndim-1)).clamp_min(1e-12)
        if self.kind=='transport':
            ga=self.scale*(B.T[None]@avg);gb=self.scale*(avg@A.T[None]);return ga,gb
        return avg[:,:A.numel()].reshape(-1,*A.shape),avg[:,A.numel():].reshape(-1,*B.shape)

    @torch.no_grad()
    def observe(self,rows,A,B):
        self.mass.mul_(self.forget);self.total.mul_(self.forget)
        if self.kind=='transport':
            # Aggregate by class before multiplying: no per-example full C table.
            for c in sorted({self.label_list[r['index']] for r in rows}):
                rr=[r for r in rows if self.label_list[r['index']]==c]
                x=torch.cat([r['x'].float() for r in rr]);g=torch.cat([r['g'].float() for r in rr])
                self.total[c].add_(g.T@x);self.mass[c]+=len(rr)
        else:
            d=pack(rows);ga,gb=raw(d['x'],d['g'],A,B,self.scale);target=torch.cat([ga.flatten(1),gb.flatten(1)],1)
            labels=self.labels[[r['index'] for r in rows]];self.total.index_add_(0,labels,target);self.mass.index_add_(0,labels,torch.ones(len(rows),device='cuda'))
        assert torch.isfinite(self.total).all()

    def step(self,m,tok,mod,cap,ds,ids,positions,opt,job,step):
        A,B,s=weights(mod);tick=clock();k=len(positions);n=sum(ds[i]['n'] for i in ids)
        burnin=step<=job.get('burnin_steps',0)
        if not burnin:ga,gb=self.predict(A,B);labels=self.labels[ids];pred=(ga[labels],gb[labels])
        prediction_seconds=clock()-tick
        subset=[ids[i] for i in positions];t=clock();rows,truth,_=collect_gpu(m,tok,mod,cap,ds,subset,micro=job.get('micro',32),keep=True,verify_reconstruction=False);label_seconds=clock()-t
        with torch.no_grad():
            gradients=tuple(t*(32/k)/n for t in truth) if burnin else tuple(((32/k)*t+job.get('coefficient',1.)*(p.sum(0)-(32/k)*p[positions].sum(0)))/n for p,t in zip(pred,truth))
        # All targets are pre-update. Observation happens after predictions and
        # the current correction draw; it only affects subsequent estimates.
        t=clock()
        if not job.get('freeze_after_burnin',False) or burnin:self.observe(rows,A,B)
        cal_seconds=clock()-t
        opt.zero_grad(set_to_none=True)
        for w,g in zip([A,B],gradients):w.grad=g.detach()
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        return dict(step=step,online_seconds=clock()-tick,prefix_prediction_seconds=prediction_seconds,label_seconds=label_seconds,calibration_seconds=cal_seconds,true_gradient_examples=k,gradient_norm=norm)
