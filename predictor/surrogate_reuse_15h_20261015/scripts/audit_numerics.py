from research import *
from correction4 import corrected_pair

def main():
    task='paws';warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter']);cap=Capture(mod);A,B,s=weights(mod)
    ds=Dataset(task,'update',tok);ids=list(range(32))
    rows,truth,reconstruction=collect(m,tok,mod,cap,ds,ids,keep=True)
    p,_=load_predictor(predictor_path(task,123));pred=per_example_predictions(p,rows,A,B,s)
    n=sum(ds[i]['n'] for i in ids);corrected=corrected_pair(pred,truth,list(range(32)),n)
    full=tuple(t/n for t in truth);k32=measures(flat(corrected),flat(full))
    cached=[{k:(v.cuda() if torch.is_tensor(v) else v) for k,v in r.items() if k not in ['g','y']} for r in rows]
    direct=current_features(mod,cached,ids)
    errors=[measures(a['y'].cpu(),b['y'].cpu()) for a,b in zip(direct,rows)]
    before_y=max(e['relative_l2'] for e in errors)
    # Change only LoRA, then compare recomputed cached-prefix Y against a full prefix.
    with torch.no_grad():B.add_(torch.randn_like(B)*.0001)
    actual=features(m,tok,mod,cap,ds,ids,micro=4)
    reused=current_features(mod,cached,ids)
    after_y=max(measures(a['y'].cpu(),b['y'].cpu())['relative_l2'] for a,b in zip(actual,reused))
    # Exhaustive 2-of-5 subsets verify the CV estimator's expectation for arbitrary lambda.
    import itertools
    torch.manual_seed(42);pp=torch.randn(5,7,device='cuda');tt=torch.randn(5,7,device='cuda')
    expected=tt.sum(0)/31;maxbias=0.
    for lam in [0.,.25,.5,1.,1.5]:
        estimates=[]
        for indices in itertools.combinations(range(5),2):
            pos=list(indices);estimates.append((2.5*tt[pos].sum(0)+lam*(pp.sum(0)-2.5*pp[pos].sum(0)))/31)
        maxbias=max(maxbias,float((torch.stack(estimates).mean(0)-expected).abs().max()))
    result=dict(reconstruction=reconstruction,cv32_vs_full=k32,cached_y_before_max_relative_l2=before_y,cached_y_after_max_relative_l2=after_y,exhaustive_cv_max_abs_bias=maxbias)
    assert reconstruction['min_reconstruction_cosine']>.999
    assert reconstruction['max_reconstruction_relative_l2']<.02
    assert k32['relative_l2']<1e-5
    assert before_y<.02 and after_y<.02
    assert maxbias<1e-6
    result['passed']=True;write(EXP/'results/numerical_audit.json',result);log(result);cap.close()

if __name__=='__main__':main()
