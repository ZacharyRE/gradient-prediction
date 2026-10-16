"""CPU-only predictor_dev gradient-variation diagnosis; no final inputs/readouts."""
from predictor_core import *
import argparse


def pack_cpu(rows):
    length=max(len(r['x']) for r in rows);n=len(rows)
    d={k:torch.zeros(n,length,rows[0][k].shape[-1],dtype=rows[0][k].dtype) for k in ['x','y','g']}
    for k in ['active','valid']:d[k]=torch.zeros(n,length,dtype=torch.bool)
    if 'label_ids' in rows[0]:d['label_ids']=torch.full((n,length),-100,dtype=torch.long)
    d['pos']=torch.zeros(n,length)
    for j,row in enumerate(rows):
        l=len(row['x'])
        for k in ['x','y','g','active']:d[k][j,:l]=row[k]
        if 'label_ids' in d:d['label_ids'][j,:l]=row['label_ids']
        d['valid'][j,:l]=True;d['pos'][j,:l]=torch.arange(l)/max(1,l-1)
    return d


def covariance_summary(p,t):
    p=p.double();t=t.double();pc=p-p.mean(0);tc=t-t.mean(0)
    pv=float(pc.square().sum());tv=float(tc.square().sum());cov=float((pc*tc).sum())
    assert pv>0 and tv>0
    return dict(centered_correlation=cov/(pv*tv)**.5,
        optimal_scalar_in_sample=cov/pv,
        cv1_centered_residual_variance_ratio=float((tc-pc).square().sum())/tv,
        prediction_mean_energy_fraction=float(p.mean(0).square().sum()*len(p)/p.square().sum()),
        truth_mean_energy_fraction=float(t.mean(0).square().sum()*len(t)/t.square().sum()),
        aggregate=measures(p.sum(0),t.sum(0)))


@torch.no_grad()
def diagnose_checkpoint(task,predictor_seed,variant=None):
    suffix='' if variant is None else '_'+variant
    directory=EXP/'models'/f'{task}_predictor{suffix}_p{predictor_seed}'
    path=directory/'best.pt';obj=torch.load(path,map_location='cpu',weights_only=False)
    predictor=Predictor(**obj['config']);predictor.load_state_dict(obj['state']);predictor.eval()
    assert predictor.input_field=='y','This CPU diagnostic supports Y-based predictors, with optional explicit labels'
    cache=EXP/'cache'/task;rows=load_samples(cache/'predictor_dev')
    metadata=torch.load(cache/'metadata.pt',map_location='cpu',weights_only=False)
    A,B,s=metadata['A'],metadata['B'],metadata['scale'];pp=[];tt=[];tick=time.time()
    for indices in sample_chunks(rows,max_n=4,budget=2048):
        d=pack_cpu([rows[i] for i in indices]);pa,pb=raw(d['x'],predictor(d),A,B,s);ta,tb=raw(d['x'],d['g'],A,B,s)
        pp.append(torch.cat([pa.flatten(1),pb.flatten(1)],1));tt.append(torch.cat([ta.flatten(1),tb.flatten(1)],1))
    p=torch.cat(pp);t=torch.cat(tt);sizes={'all':slice(None),'factor_A':slice(0,A.numel()),'factor_B':slice(A.numel(),None)}
    # Internal source-group split of predictor_dev; neither accuracy_dev nor
    # gradient_test is consulted, and these coefficients are not deployed.
    source=read(data(task,'predictor_dev'))
    from analyze import group_key
    groups=[group_key(task,source[r['index']]) for r in rows]
    unique=sorted(set(groups));rng=np.random.default_rng(20261010);rng.shuffle(unique)
    first=set(unique[:len(unique)//2]);left=torch.tensor([g in first for g in groups]);right=~left
    result={}
    for label,section in sizes.items():
        pred=p[:,section];truth=t[:,section];r=covariance_summary(pred,truth)
        folds=[]
        for fit,check in [(left,right),(right,left)]:
            scalar=covariance_summary(pred[fit],truth[fit])['optimal_scalar_in_sample']
            scalar=max(0,min(4,scalar))
            pc=pred[check].double()-pred[check].double().mean(0);tc=truth[check].double()-truth[check].double().mean(0)
            folds.append(dict(fit_n=int(fit.sum()),check_n=int(check.sum()),coefficient=scalar,
                residual_variance_ratio=float((tc-scalar*pc).square().sum()/tc.square().sum())))
        r['internal_group_split_coefficient_diagnostic']=folds;result[label]=r
    assert not torch.cuda.is_initialized()
    return dict(task=task,predictor_seed=predictor_seed,variant=variant,label_dim=predictor.label_dim,checkpoint_sha256=sha(path),
        predictor_dev_manifest_sha256=sha(cache/'predictor_dev/manifest.json'),n=len(rows),source_groups=len(unique),
        seconds=time.time()-tick,metrics=result,
        interpretation='CPU FP32 inference on the existing predictor_dev cache at fixed warmup. Centered raw-gradient variance approximates the control-variate variance question, not Adam updates or downstream accuracy. Internal group-split coefficients are diagnostics only; no recipe is changed by this script.',
        full_sequences=True,cuda_initialized=False,final_test_readout_used=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',default=['boolq','mnli','paws']);a=ap.parse_args()
    torch.set_num_threads(4);results=[]
    for task in a.tasks:
        for variant in ([None,'batchfit'] if task=='boolq' else [None]):
            r=diagnose_checkpoint(task,123,variant);results.append(r)
            write(EXP/'results/predictor_gradient_variation_diagnostic.json',dict(time=time.time(),records=results,complete=False,final_test_readout_used=False))
            log(dict(task=task,variant=variant,seconds=r['seconds'],all=r['metrics']['all']))
    write(EXP/'results/predictor_gradient_variation_diagnostic.json',dict(time=time.time(),records=results,complete=True,final_test_readout_used=False))


if __name__=='__main__':main()
