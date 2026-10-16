"""Surrogate plus Bernoulli-query true-gradient correction.

Given a pre-query estimate b, use b+I/q*(g-b), E[I]=q. This is unbiased
before clipping/Adam. A delayed EMA of observed residuals improves b. The
first step is a guaranteed, unscaled true gradient. Matched no-predictor
controls retain the same query schedule and residual estimator.
"""
import research_fastio as core
from research_fastio import *
STATE={}

def queried_update(m,tok,mod,cap,p,ds,ids,positions,opt,job,step,replay,calopt,calrng,lambda_state,cached):
    A,B,s=weights(mod);tick=clock();n=sum(ds[i]['n'] for i in ids)
    if step==1:STATE.clear()
    row=dict(step=step,true_gradient_examples=0,calibration_seconds=0.,prefix_prediction_seconds=0.,label_seconds=0.,unbiased_raw_query_estimator=True)
    # Form b before drawing the query, using the SAME feature path in both cases.
    if p is not None:
        start=clock();rr=current_features(mod,cached,ids)
        with torch.no_grad():pa,pb=per_example_predictions(p,rr,A,B,s)
        pred=(pa.sum(0)/n,pb.sum(0)/n);row['prefix_prediction_seconds']=clock()-start
    else:pred=(torch.zeros_like(A),torch.zeros_like(B))
    previous=STATE.get('residual',(torch.zeros_like(A),torch.zeros_like(B)))
    base=tuple(pp+rr for pp,rr in zip(pred,previous))
    q=job['query_probability'];assert 0<q<=1
    queried=step==1 or calrng.random()<q
    if queried:
        start=clock();_,rawtruth,_=collect(m,tok,mod,cap,ds,ids,micro=32,keep=False,verify_reconstruction=False)
        truth=tuple(t/n for t in rawtruth);row['label_seconds']=clock()-start;row['true_gradient_examples']=len(ids)
        with torch.no_grad():
            gradients=truth if step==1 else tuple(b+(t-b)/q for b,t in zip(base,truth))
            residual=tuple(t-p0 for t,p0 in zip(truth,pred));beta=job.get('residual_ema',.9)
            STATE['residual']=residual if step==1 else tuple(beta*a+(1-beta)*b for a,b in zip(previous,residual))
    else:gradients=base
    opt.zero_grad(set_to_none=True)
    for w,g in zip([A,B],gradients):
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite Bernoulli-query gradient')
        w.grad=g.detach()
    row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
    row.update(queried=queried,query_probability=q,online_seconds=clock()-tick)
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    core.update=queried_update;core.ENTRY_SHA=sha(Path(__file__));core.CODE_SHA=source_signature(Path(__file__))
    core.run(a.task,json.loads(Path(a.jobs).read_text()),a.seed,a.predictor_seed)

if __name__=='__main__':main()
