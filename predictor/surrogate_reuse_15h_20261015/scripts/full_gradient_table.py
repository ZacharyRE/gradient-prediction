"""Full-data control variates with an incrementally maintained gradient memory.

Unlike the 32-example minibatch table control, the predictor mean covers the
entire candidate dataset. IID subset sampling makes the raw estimator unbiased
for its full-data token-mean gradient. Clipping and Adam remain nonlinear.
"""
from predictor_core import *
from gpu_features import collect_gpu
from updates_legacy import clock

class FullGradientTable:
    def __init__(self,m,tok,mod,cap,ds,cached,job):
        assert job.get('dataset_sampling')=='iid','Full-data unbiased correction requires independent population sampling'
        self.ds=ds;self.cached=cached;self.rows=[None]*len(ds);self.ages=[None]*len(ds);self.known=0;self.bootstrap_examples=0;self.mode=job.get('table_mode','activation');A,B,s=weights(mod)
        self.total_tokens=sum(ds[i]['n'] for i in range(len(ds)))
        self.total=torch.zeros(B.shape[0],A.shape[1],device=A.device,dtype=torch.float64) if self.mode=='activation' else [torch.zeros_like(A,dtype=torch.float64),torch.zeros_like(B,dtype=torch.float64)]
        if job.get('table_bootstrap',True):
            for ids in chunks(ds,list(range(len(ds))),max_n=32,token_budget=8192):
                rows,_,_=collect_gpu(m,tok,mod,cap,ds,ids,micro=32,keep=True,verify_reconstruction=False);self.observe(rows,mod,0);self.bootstrap_examples+=len(ids)

    @torch.no_grad()
    def observe(self,rows,mod,step):
        A,B,s=weights(mod)
        for r in rows:
            i=r['index'];old=self.rows[i]
            if self.mode=='activation':
                new=r['g'].bfloat16().clone();delta=new.float() if old is None else new.float()-old.float();self.total.add_((delta.T@r['x'].float()).double())
            else:
                new=tuple(v[0].bfloat16().clone() for v in raw(r['x'][None],r['g'][None],A,B,s))
                for j,v in enumerate(new):self.total[j].add_(v.double() if old is None else v.double()-old[j].double())
            if old is None:self.known+=1
            self.rows[i]=new;self.ages[i]=step

    @torch.no_grad()
    def sum_prediction(self,mod):
        A,B,s=weights(mod)
        if self.mode=='activation':C=self.total.float();return s*B.T@C,s*C@A.T
        return tuple(v.float() for v in self.total)

    @torch.no_grad()
    def predict(self,ids,mod):
        A,B,s=weights(mod);result=[torch.zeros_like(A),torch.zeros_like(B)]
        for i in ids:
            if self.rows[i] is None:continue
            pair=raw(self.cached[i]['x'][None],self.rows[i][None],A,B,s) if self.mode=='activation' else [v[None] for v in self.rows[i]]
            for acc,v in zip(result,pair):acc.add_(v[0])
        return result

    def step(self,m,tok,mod,cap,ds,ids,positions,opt,job,step):
        tick=clock();A,B,s=weights(mod);selected=[ids[i] for i in positions];k=len(selected)
        t=clock();total=self.sum_prediction(mod);pred=self.predict(selected,mod);prediction_seconds=clock()-t
        t=clock();rows,truth,_=collect_gpu(m,tok,mod,cap,ds,selected,micro=32,keep=True,verify_reconstruction=False);label_seconds=clock()-t
        gradients=tuple((p+(len(ds)/k)*(g-q))/self.total_tokens for p,g,q in zip(total,truth,pred));ages=[step-self.ages[i] for i in selected if self.ages[i] is not None]
        t=clock();self.observe(rows,mod,step);memory_seconds=clock()-t
        opt.zero_grad(set_to_none=True)
        for w,g in zip([A,B],gradients):assert torch.isfinite(g).all();w.grad=g.detach()
        norm=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
        return dict(step=step,online_seconds=clock()-tick,true_gradient_examples=k,calibration_seconds=0.,gradient_norm=norm,prefix_prediction_seconds=prediction_seconds,label_seconds=label_seconds,memory_update_seconds=memory_seconds,table_known=self.known,table_fraction=self.known/len(ds),mean_selected_prediction_age=float(np.mean(ages)) if ages else None,estimator='Full-dataset token-mean CV under iid population sampling, unbiased before numerical rounding/clipping/Adam',table_mode=self.mode)
