"""Periodic exact-gradient anchoring with a remembered prediction residual.

Unlike fresh CV correction, updates between anchors are BIASED approximations.
Matched no-predictor gradient-hold controls are therefore essential.
"""
import research_fastio as core
from research_fastio import *
STATE={}

def anchored_update(m,tok,mod,cap,p,ds,ids,positions,opt,job,step,replay,calopt,calrng,lambda_state,cached):
    A,B,s=weights(mod);tick=clock();n=sum(ds[i]['n'] for i in ids)
    if step==1:STATE.clear()
    anchor=(step-1)%job['anchor_every']==0
    row=dict(step=step,anchor=anchor,true_gradient_examples=0,calibration_seconds=0.,prefix_prediction_seconds=0.,label_seconds=0.,biased_between_anchors=True)
    labels=None;truth=None
    if anchor:
        t=clock();labels,rawtruth,_=collect(m,tok,mod,cap,ds,ids,micro=job.get('micro',32),keep=p is not None,verify_reconstruction=False)
        truth=tuple(v/n for v in rawtruth);row['label_seconds']=clock()-t;row['true_gradient_examples']=len(ids)
        if job.get('cal_on_refresh'):
            row['calibration']=train_calibration(p,calopt,labels,A.detach(),B.detach(),s,job,calrng)
            row['calibration_seconds']=row['calibration']['seconds']
    if p is not None:
        t=clock()
        rr=labels if anchor else (current_features(mod,cached,ids) if cached is not None else features(m,tok,mod,cap,ds,ids,micro=job.get('prefix_micro',16)))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=job.get('pred_bf16',False)):
            pa,pb=per_example_predictions(p,rr,A,B,s)
            pred=(pa.sum(0)/n,pb.sum(0)/n)
        row['prefix_prediction_seconds']=clock()-t
    else:pred=(torch.zeros_like(A),torch.zeros_like(B))
    with torch.no_grad():
        if anchor:
            residual=tuple(t-p0 for t,p0 in zip(truth,pred))
            beta=job.get('residual_ema',0.)
            STATE['residual']=residual if 'residual' not in STATE else tuple(beta*a+(1-beta)*b for a,b in zip(STATE['residual'],residual))
            gradients=truth
        else:gradients=tuple(p0+r for p0,r in zip(pred,STATE['residual']))
    opt.zero_grad(set_to_none=True)
    for w,g in zip([A,B],gradients):
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite anchor estimate')
        w.grad=g.detach()
    row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
    row['online_seconds']=clock()-tick
    return row

ORIGINAL_AUDIT=core.variance_diagnostic
def anchor_audit(p,rows,A,B,s,coefficients):
    result=ORIGINAL_AUDIT(p,rows,A,B,s,coefficients)
    with torch.no_grad():
        pp=per_example_predictions(p,rows,A,B,s)
        truth=individual_truth_for_audit(rows,A,B,s)
        n=sum(r['n'] for r in rows)
        adjusted=flat(tuple(v.sum(0)/n+r for v,r in zip(pp,STATE['residual'])))
        result['residual_adjusted_batch']=measures(adjusted,truth/n)
    return result

def individual_truth_for_audit(rows,A,B,s):
    d=pack(rows);ta,tb=raw(d['x'],d['g'],A,B,s)
    return flat((ta.sum(0),tb.sum(0)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    core.update=anchored_update;core.variance_diagnostic=anchor_audit;core.ENTRY_SHA=sha(Path(__file__));core.CODE_SHA=source_signature(Path(__file__))
    core.run(a.task,json.loads(Path(a.jobs).read_text()),a.seed,a.predictor_seed)

if __name__=='__main__':main()
