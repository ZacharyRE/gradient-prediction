"""Prefix capture without round-tripping activation tensors through CPU memory."""
import predictor_core as pc
from predictor_core import torch
import time

@torch.no_grad()
def features_gpu(m,tok,mod,cap,ds,ids,micro=32):
    samples=[]
    class PrefixDone(Exception):pass
    def stop(*args):raise PrefixDone()
    handle=mod.register_forward_hook(stop) if cap.residual_norm is None else cap.residual_norm.register_forward_pre_hook(stop)
    try:
        for ix in pc.chunks(ds,ids,max_n=micro):
            bb=pc.batch(ds,ix,tok)
            try:pc.forward(m,bb)
            except PrefixDone:pass
            for j,i in enumerate(ix):
                length=len(ds[i]['input_ids']);labels=torch.tensor(ds[i]['labels'][1:]+[-100],device='cuda')
                row=dict(index=i,x=cap.x[j,:length].detach().bfloat16(),y=cap.y[j,:length].detach().bfloat16(),active=labels>=0,n=ds[i]['n'],label_ids=labels)
                if cap.residual_norm is not None:row['h']=cap.h[j,:length].detach().bfloat16()
                samples.append(row)
    finally:handle.remove()
    return samples

def collect_gpu(m,tok,mod,cap,ds,ids,micro=4,keep=True,verify_reconstruction=True):
    if not keep:return pc.collect(m,tok,mod,cap,ds,ids,micro=micro,keep=False,verify_reconstruction=verify_reconstruction)
    A,B,s=pc.weights(mod);truth=[torch.zeros_like(A),torch.zeros_like(B)];samples=[];audits=[];started=time.time()
    for ix in pc.chunks(ds,ids,max_n=micro):
        bb=pc.batch(ds,ix,tok);m.zero_grad(set_to_none=True);n=sum(ds[i]['n'] for i in ix)
        loss=pc.forward(m,bb).loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher loss')
        (loss*n).backward()
        if not all(torch.isfinite(v.grad).all() for v in [A,B]):raise FloatingPointError('Nonfinite teacher gradient')
        truth[0]+=A.grad;truth[1]+=B.grad
        x=cap.x.detach();y=cap.y.detach();g=cap.y.grad.detach()
        if verify_reconstruction:
            with torch.no_grad():ga,gb=pc.raw(x,g,A,B,s)
            audits.append(pc.measures(pc.flat((ga.sum(0),gb.sum(0))),pc.flat((A.grad,B.grad))))
        for j,i in enumerate(ix):
            length=len(ds[i]['input_ids']);labels=torch.tensor(ds[i]['labels'][1:]+[-100],device='cuda')
            row=dict(index=i,x=x[j,:length].bfloat16(),y=y[j,:length].bfloat16(),g=g[j,:length].float(),active=labels>=0,n=ds[i]['n'],label_ids=labels)
            if cap.residual_norm is not None:row['h']=cap.h[j,:length].detach().bfloat16()
            samples.append(row)
        del loss,bb
    m.zero_grad(set_to_none=True)
    return samples,truth,dict(seconds=time.time()-started,max_reconstruction_relative_l2=max((r['relative_l2'] for r in audits),default=0),min_reconstruction_cosine=min((r['cosine'] for r in audits),default=1))
