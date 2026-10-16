"""Delayed per-example gradient memory used as a minibatch control variate.

Inspired by incremental-gradient tables, but this is not classical SAGA:
we correct a 32-example batch estimator and then apply clipping plus Adam.
Activation mode transports stored suffix derivatives through current A/B.
No claim of the original convex convergence guarantee applies.
"""
from predictor_core import *
from gpu_features import collect_gpu
from updates_legacy import clock

class GradientTable:
    def __init__(self,m,tok,mod,cap,ds,cached,job):
        self.rows=[None]*len(ds);self.ages=[None]*len(ds);self.cached=cached
        self.mode=job.get('table_mode','activation');self.coefs=[1.,1.];self.bootstrap_examples=0
        self.dtype=torch.bfloat16;self.bytes=0;self.known=0
        if job.get('table_bootstrap',False):
            for ids in chunks(ds,list(range(len(ds))),max_n=32,token_budget=8192):
                rows,_,_=collect_gpu(m,tok,mod,cap,ds,ids,micro=32,keep=True,verify_reconstruction=False)
                self.observe(rows,mod,0);self.bootstrap_examples+=len(ids)

    @torch.no_grad()
    def observe(self,rows,mod,step):
        A,B,s=weights(mod)
        for r in rows:
            i=r['index']
            if self.mode=='activation':value=r['g'].to(self.dtype).clone()
            elif self.mode=='factor':value=tuple(v.to(self.dtype).clone() for v in raw(r['x'][None],r['g'][None],A,B,s))
            else:raise ValueError(self.mode)
            if self.rows[i] is None:
                self.known+=1;self.bytes+=sum(v.numel()*v.element_size() for v in value) if isinstance(value,tuple) else value.numel()*value.element_size()
            self.rows[i]=value;self.ages[i]=step

    @torch.no_grad()
    def predict(self,ids,mod):
        A,B,s=weights(mod);pa=torch.zeros(len(ids),*A.shape,device=A.device);pb=torch.zeros(len(ids),*B.shape,device=B.device)
        if self.mode=='factor':
            for j,i in enumerate(ids):
                if self.rows[i] is not None:pa[j]=self.rows[i][0][0];pb[j]=self.rows[i][1][0]
        else:
            known=[j for j,i in enumerate(ids) if self.rows[i] is not None]
            for start in range(0,len(known),16):
                ix=known[start:start+16];L=max(len(self.cached[ids[j]]['x']) for j in ix)
                x=torch.zeros(len(ix),L,A.shape[1],device=A.device,dtype=torch.bfloat16);g=torch.zeros_like(x)
                for b,j in enumerate(ix):
                    i=ids[j];length=len(self.rows[i]);x[b,:length]=self.cached[i]['x'];g[b,:length]=self.rows[i]
                ga,gb=raw(x,g,A,B,s);pa[ix]=ga;pb[ix]=gb
        return pa,pb

    def step(self,m,tok,mod,cap,ds,ids,positions,opt,job,step):
        tick=clock();A,B,s=weights(mod);n=sum(ds[i]['n'] for i in ids);k=len(positions)
        refresh=bool(job.get('refresh_every') and step%job['refresh_every']==0)
        if refresh:positions=list(range(len(ids)));k=len(ids)
        coefficients=self.coefs.copy() if job.get('lambda_mode')=='ema' else [job.get('coefficient',1.)]*2
        t=clock();pred=self.predict(ids,mod) if not refresh else None;prediction_seconds=clock()-t
        t=clock();labels,truth,_=collect_gpu(m,tok,mod,cap,ds,[ids[j] for j in positions],micro=job.get('micro',32),keep=True,verify_reconstruction=False);label_seconds=clock()-t
        with torch.no_grad():
            if refresh:gradients=tuple(t/n for t in truth)
            else:gradients=tuple(((len(ids)/k)*t+c*(p.sum(0)-(len(ids)/k)*p[positions].sum(0)))/n for p,t,c in zip(pred,truth,coefficients))
            if job.get('lambda_mode')=='ema' and not refresh:
                d=pack(labels);ta,tb=raw(d['x'],d['g'],A,B,s)
                for j,(p,t) in enumerate(zip(pred,[ta,tb])):
                    p=p[positions].flatten(1);t=t.flatten(1);p=p-p.mean(0);t=t-t.mean(0)
                    value=float((p*t).sum()/p.square().sum().clamp_min(1e-20));value=float(np.clip(value,0,job.get('lambda_max',2.)))
                    self.coefs[j]=.9*self.coefs[j]+.1*value
        # The table and coefficients are changed only after this estimator is fixed.
        ages=[step-self.ages[i] for i in ids if self.ages[i] is not None]
        t=clock();self.observe(labels,mod,step);memory_update_seconds=clock()-t
        opt.zero_grad(set_to_none=True)
        for w,g in zip([A,B],gradients):
            assert torch.isfinite(g).all();w.grad=g.detach()
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        return dict(step=step,online_seconds=clock()-tick,true_gradient_examples=k,calibration_seconds=0.,prefix_prediction_seconds=prediction_seconds,label_seconds=label_seconds,memory_update_seconds=memory_update_seconds,gradient_norm=norm,coefficients=coefficients,next_coefficients=self.coefs.copy(),refresh=refresh,table_known=self.known,table_fraction=self.known/len(ds),table_bytes=self.bytes,mean_prediction_age=float(np.mean(ages)) if ages else None)
