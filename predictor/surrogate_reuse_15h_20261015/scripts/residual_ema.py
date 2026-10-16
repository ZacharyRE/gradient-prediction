"""Biased low-variance updates: predict the batch, smooth its correction over time.

The matched no-predictor control smooths true subset gradients in exactly the
same way. One initial full batch initializes both. This is not unbiased CV.
"""
import research_v3 as core
from research_v3 import *
STATE={}

def update_smoothed(m,tok,mod,cap,p,ds,ids,positions,opt,job,step,replay,calopt,calrng,lambda_state,cached):
    A,B,s=weights(mod);tick=clock();n=sum(ds[i]['n'] for i in ids)
    if step==1:STATE.clear();positions=list(range(len(ids)))
    row=dict(step=step,true_gradient_examples=len(positions),calibration_seconds=0.,prefix_prediction_seconds=0.,label_seconds=0.,biased_temporal_correction=True)
    if p is not None:
        t=clock();rr=current_features(mod,cached,ids);pred=per_example_predictions(p,rr,A,B,s)
        row['prefix_prediction_seconds']=clock()-t
    else:pred=tuple(torch.zeros(len(ids),*w.shape,device=w.device) for w in [A,B])
    t=clock();_,truth,_=collect(m,tok,mod,cap,ds,[ids[i] for i in positions],micro=32,keep=False,verify_reconstruction=False)
    row['label_seconds']=clock()-t
    with torch.no_grad():
        residual=tuple(len(ids)/len(positions)*(t-pp[positions].sum(0))/n for t,pp in zip(truth,pred))
        old=STATE.get('residual',residual);alpha=job.get('current_mix',0.)
        gradients=tuple(pp.sum(0)/n+alpha*r+(1-alpha)*o for pp,r,o in zip(pred,residual,old))
        beta=job.get('residual_beta',.9)
        STATE['residual']=tuple(beta*o+(1-beta)*r for o,r in zip(old,residual))
    opt.zero_grad(set_to_none=True)
    for w,g in zip([A,B],gradients):w.grad=g.detach()
    row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
    row['online_seconds']=clock()-tick
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    core.update=update_smoothed;core.ENTRY_SHA=sha(Path(__file__));core.CODE_SHA=source_signature(Path(__file__))
    core.run(a.task,json.loads(Path(a.jobs).read_text()),a.seed,a.predictor_seed)

if __name__=='__main__':main()
