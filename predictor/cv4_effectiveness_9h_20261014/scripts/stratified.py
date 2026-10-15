"""Label-stratified correction with inverse-inclusion weights.

Training labels are known before sampling. Each present class receives at least
one uniformly sampled correction example. Both subset-only and neural CV use
the same reconstructed per-example gradients, avoiding a mismatched control.
"""
import research_v3 as core
from research_v3 import *

def choose(groups,k,rng):
    keys=sorted(groups);counts={g:1 for g in keys}
    assert k>=len(keys)
    for _ in range(k-len(keys)):
        eligible=[g for g in keys if counts[g]<len(groups[g])]
        g=max(eligible,key=lambda g:len(groups[g])/counts[g])
        counts[g]+=1
    positions=[];weights=[]
    for g in keys:
        positions.extend(map(int,rng.choice(groups[g],counts[g],replace=False)))
        weights.extend([len(groups[g])/counts[g]]*counts[g])
    return positions,weights

def stratified_update(m,tok,mod,cap,p,ds,ids,positions,opt,job,step,replay,calopt,calrng,lambda_state,cached):
    A,B,s=weights(mod);tick=clock();n=sum(ds[i]['n'] for i in ids)
    groups={}
    for j,i in enumerate(ids):groups.setdefault(ds.rows[i]['gold'],[]).append(j)
    if job.get('stratified',True):
        positions,iw=choose(groups,job['k'],np.random.default_rng(job.get('sampling_seed',10000)+step))
    else:iw=[len(ids)/len(positions)]*len(positions)
    row=dict(step=step,true_gradient_examples=len(positions),calibration_seconds=0.,prefix_prediction_seconds=0.,label_seconds=0.,correction_positions=positions,inverse_inclusion_weights=iw,stratified_sampling=job.get('stratified',True))
    pred=None
    if p is not None:
        t=clock();rr=current_features(mod,cached,ids)
        pred=per_example_predictions(p,rr,A,B,s)
        row['prefix_prediction_seconds']=clock()-t
    t=clock();labels,_,_=collect(m,tok,mod,cap,ds,[ids[i] for i in positions],micro=job['k'],keep=True,verify_reconstruction=False)
    row['label_seconds']=clock()-t
    with torch.no_grad():
        d=pack(labels);truth=raw(d['x'],d['g'],A,B,s)
        w=torch.tensor(iw,device='cuda')[:,None,None]
        if pred is None:gradients=tuple((v*w).sum(0)/n for v in truth)
        else:
            c=job.get('coefficient',1.)
            gradients=tuple((c*pp.sum(0)+((tt-c*pp[positions])*w).sum(0))/n for pp,tt in zip(pred,truth))
    oldA=A.detach().clone() if job.get('cal_every') else None
    oldB=B.detach().clone() if job.get('cal_every') else None
    opt.zero_grad(set_to_none=True)
    for parameter,g in zip([A,B],gradients):parameter.grad=g
    row['gradient_norm']=float(nn.utils.clip_grad_norm_([A,B],1.,error_if_nonfinite=True));opt.step();m.zero_grad(set_to_none=True)
    if job.get('cal_every'):
        replay.extend(labels);del replay[:-job.get('replay_size',32)]
        if step%job['cal_every']==0:
            row['calibration']=train_calibration(p,calopt,replay,oldA,oldB,s,job,calrng)
            row['calibration_seconds']=row['calibration']['seconds']
    row['online_seconds']=clock()-tick
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    jobs=json.loads(Path(a.jobs).read_text())
    for j in jobs:j['sampling_seed']=a.seed+10000
    core.update=stratified_update;core.ENTRY_SHA=sha(Path(__file__));core.CODE_SHA=source_signature(Path(__file__))
    core.run(a.task,jobs,a.seed,a.predictor_seed)

if __name__=='__main__':main()
