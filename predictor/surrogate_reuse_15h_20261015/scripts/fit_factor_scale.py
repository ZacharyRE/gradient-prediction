"""Closed-form scalar-template fitting; PCA and regressors use train rows only."""
from predictor_core import *
from factor_scale import FactorScale
from fit_factor import ratio
from updates_legacy import clock

@torch.no_grad()
def main():
    gpu_guard();guard_space();seed(941);started=clock();source=EXP/'cache/factor_v1';manifest=json.loads((source/'complete.json').read_text());base_path=EXP/'models/mnli_factor_u128v256/best.pt';base=torch.load(base_path,map_location='cuda',weights_only=False);assert base['epoch']==0
    raw_states=[dict(name=x['name'],**torch.load(source/x['file'],map_location='cpu',weights_only=False)) for x in manifest['states']];all_features=torch.cat([r['train']['features'] for r in raw_states]).cuda();mean=all_features.mean(0);centered=all_features-mean;_,_,basis=torch.pca_lowrank(centered,q=128,center=False,niter=3);std=(centered@basis).std(0).clamp_min(.01);pca_seconds=clock()-started
    prepared=[]
    for r in raw_states:
        A=r['meta']['A'].cuda();B=r['meta']['B'].cuda();s=r['meta']['scale'];st=dict(name=r['name'],A=A,B=B,s=s)
        for split in ['train','dev']:
            d={k:v.cuda() for k,v in r[split].items() if k!='coefficients'};C=base['state']['class_mean'][d['classes']];pa=s*((B.T@base['state']['U'])[None]@C@base['state']['V'].T);pb=s*(base['state']['U'][None]@C@(base['state']['V'].T@A.T));energy=torch.stack([v.square().sum((1,2)) for v in [pa,pb]],1).clamp_min(1e-20);dot=torch.stack([(v*t).sum((1,2)) for v,t in zip([pa,pb],[d['ga'],d['gb']])],1);normalizers=torch.stack([d['ga'].square().sum((1,2)).mean(),d['gb'].square().sum((1,2)).mean()]).clamp_min(1e-20)
            d.update(target_scale=dot/energy,regression_weight=energy/normalizers)
            if split=='train':del d['ga'],d['gb']
            else:d.update(pa=pa,pb=pb)
            st[split]=d
        prepared.append(st)
    del raw_states,all_features,centered;reserve_worker_cache();reports=[]
    for q in [32,64,128]:
        out=EXP/'models'/f'mnli_factor_scale_q{q}';out.mkdir(exist_ok=True);p=FactorScale(1536,q=q,class_tokens=base['config']['class_tokens']).cuda()
        for name in ['U','V','class_mean']:getattr(p,name).copy_(base['state'][name])
        p.feature_mean.copy_(mean);p.feature_basis.copy_(basis[:,:q]);p.feature_std.copy_(std[:q]);train={k:torch.cat([st['train'][k] for st in prepared]) for k in ['features','classes','target_scale','regression_weight']};X=p.design(train['features']);history=[];best=float('inf');selected=None;fit_start=clock()
        for alpha in [None,.001,.01,.1,1.,10.]:
            p.regression.zero_()
            if alpha is not None:
                for c in range(3):
                    mask=train['classes']==c;x=X[mask].double()
                    for factor in range(2):
                        w=train['regression_weight'][mask,factor].double();target=train['target_scale'][mask,factor].double()-1;system=x.T@(w[:,None]*x)/len(x)+alpha*torch.eye(x.shape[1],device=x.device,dtype=x.dtype);rhs=x.T@(w*target)/len(x);p.regression[c,:,factor]=torch.linalg.solve(system,rhs).float()
            metrics={}
            for st in prepared:
                d=st['dev'];scales=p.scales(d['features'],d['classes']);metrics[st['name']]=ratio(d['pa']*scales[:,0,None,None],d['pb']*scales[:,1,None,None],d['ga'],d['gb'])
            score=float(np.mean([m['centered_cv_ratio'] for m in metrics.values()]));history.append(dict(alpha=alpha,score=score,states=metrics))
            if score<best:best=score;selected=alpha;save_predictor(p,out/'best.pt',dev_score=best,alpha=alpha)
        oracle={}
        for st in prepared:
            d=st['dev'];scales=d['target_scale'].clamp(0,3);oracle[st['name']]=ratio(d['pa']*scales[:,0,None,None],d['pb']*scales[:,1,None,None],d['ga'],d['gb'])
        config=dict(q=q,source_predictor_sha256=sha(base_path),source_cache_sha256=sha(source/'complete.json'),source_sha256=source_signature(__file__),selection='Mean within-state centered gradient residual ratio on predictor_dev; no downstream data used',features='Training-only PCA of current H/X features, loglength and class-specific regressors',hypothesis='Predict a few bounded magnitudes instead of an entire high-dimensional gradient matrix')
        report=dict(complete=True,best_centered_cv_ratio=best,selected_ridge_alpha=selected,history=history,oracle_true_gradient_scalar_diagnostic=oracle,fit_seconds=clock()-fit_start,shared_preparation_seconds=pca_seconds,all_prior_teacher_and_template_costs_additional=True,best_sha256=sha(out/'best.pt'))
        write(out/'config.json',config);write(out/'summary.json',report);reports.append(dict(q=q,best=best,alpha=selected,scalar_oracle_ratio=float(np.mean([r['centered_cv_ratio'] for r in oracle.values()]))));log(reports[-1])
    write(EXP/'results/factor_scale_fit.json',dict(complete=True,seconds=clock()-started,results=reports))

if __name__=='__main__':main()
