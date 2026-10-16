"""Fast direct structured-gradient fitting with an exact low-dimensional loss."""
from predictor_core import *
from factor_surrogate import FactorSurrogate
from updates_legacy import clock
import argparse,gc

def ratio(pred_a,pred_b,true_a,true_b):
    pred=torch.cat([pred_a.flatten(1),pred_b.flatten(1)],1);truth=torch.cat([true_a.flatten(1),true_b.flatten(1)],1)
    pc=pred-pred.mean(0);tc=truth-truth.mean(0)
    return dict(centered_cv_ratio=float((tc-pc).square().sum()/tc.square().sum().clamp_min(1e-20)),centered_correlation=float((pc*tc).sum()/(pc.norm()*tc.norm()).clamp_min(1e-20)),relative_l2=float((pred-truth).norm()/truth.norm().clamp_min(1e-20)))

@torch.no_grad()
def evaluate(p,states,projection=False):
    result={};p.eval()
    for st in states:
        d=st['dev'];C=d['coefficients'] if projection else p.coefficients(d['features'],d['classes'])
        pa=st['s']*(st['M'][None]@C@p.V.T);pb=st['s']*(p.U[None]@C@st['N'][None])
        result[st['name']]=ratio(pa,pb,d['ga'],d['gb'])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--cache',default='factor_v1');a=ap.parse_args();gpu_guard();guard_space();seed(929)
    path=EXP/'cache'/a.cache;manifest=json.loads((path/'complete.json').read_text());basis=torch.load(path/'basis.pt',map_location='cpu',weights_only=False)
    raw_states=[dict(name=x['name'],**torch.load(path/x['file'],map_location='cpu',weights_only=False)) for x in manifest['states']]
    for job in json.loads(Path(a.jobs).read_text()):
        out=EXP/'models'/f'mnli_factor_{job["name"]}'
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True);seed(job.get('seed',929));started=clock();ru,rv=job.get('rank_u',32),job.get('rank_v',64)
        p=FactorSurrogate(basis['U'].shape[0],ru,rv,job.get('width',256),basis['class_tokens'],feature_mode=job.get('feature_mode','hx')).cuda();p.U.copy_(basis['U'][:,:ru]);p.V.copy_(basis['V'][:,:rv]);states=[]
        for r in raw_states:
            A=r['meta']['A'].cuda();B=r['meta']['B'].cuda();s=r['meta']['scale'];M=B.T@p.U;N=p.V.T@A.T
            td={k:(v[:,:ru,:rv] if k=='coefficients' else v).cuda() for k,v in r['train'].items()}
            vd={k:(v[:,:ru,:rv] if k=='coefficients' else v).cuda() for k,v in r['dev'].items()}
            if p.feature_mode=='h':
                for d in [td,vd]:d['features']=torch.cat([d['features'][:,:3*p.cfg['dim']],d['features'][:,-(len(p.class_tokens)+1):]],1)
            na=td['ga'].square().sum((1,2)).mean().clamp_min(1e-12);nb=td['gb'].square().sum((1,2)).mean().clamp_min(1e-12)
            td['cross']=(M.T[None]@(td['ga']@p.V))/na+((p.U.T[None]@td['gb'])@N.T[None])/nb
            td['constant']=td['ga'].square().sum((1,2))/na+td['gb'].square().sum((1,2))/nb
            del td['ga'],td['gb']
            states.append(dict(name=r['name'],train=td,dev=vd,A=A,B=B,s=s,M=M,N=N,gram_a=M.T@M/na,gram_b=N@N.T/nb))
        allcoef=torch.cat([st['train']['coefficients'] for st in states]);classes=torch.cat([st['train']['classes'] for st in states])
        with torch.no_grad():
            for c in range(len(p.class_tokens)):p.class_mean[c]=allcoef[classes==c].mean(0)
            scale=(allcoef-p.class_mean[classes]).square().mean(0).sqrt();p.coefficient_scale.copy_(scale.clamp_min(scale.mean()*.05))
        del allcoef,classes;reserve_worker_cache();projection=evaluate(p,states,True)
        write(out/'config.json',dict(job=job,cache_sha256=sha(path/'complete.json'),source_sha256=source_signature(__file__),projection_only_diagnostic=projection,meaning='Predict C in learned training-only U/V basis; reconstruct with CURRENT LoRA factors; class means plus neural residual; gradients use the original full-vocabulary teacher'))
        opt=torch.optim.AdamW(p.parameters(),lr=job.get('lr',.0003),weight_decay=.01);rng=np.random.default_rng(job.get('seed',929));history=[];best=float('inf');best_epoch=None
        for epoch in range(job.get('epochs',100)+1):
            if deadline(7200):raise RuntimeError('Reserve final validation')
            losses=[]
            if epoch:
                batches=[]
                for i,st in enumerate(states):
                    order=rng.permutation(len(st['train']['features']));batches.extend((i,order[j:j+64].tolist()) for j in range(0,len(order),64))
                rng.shuffle(batches);p.train()
                for i,ix in batches:
                    st=states[i];d=st['train'];opt.zero_grad(set_to_none=True);C=p.coefficients(d['features'][ix],d['classes'][ix]);s=st['s']
                    quad_a=(C*(st['gram_a'][None]@C)).sum((1,2));quad_b=(C*(C@st['gram_b'][None])).sum((1,2));cross=(C*d['cross'][ix]).sum((1,2))
                    loss=.5*(s*s*(quad_a+quad_b)-2*s*cross+d['constant'][ix]).mean()
                    loss=loss+job.get('coefficient_regularization',.01)*((C-d['coefficients'][ix])/p.coefficient_scale).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite factor loss')
                    loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            if epoch%5==0 or epoch==job.get('epochs',100):
                metrics=evaluate(p,states);value=float(np.mean([x['centered_cv_ratio'] for x in metrics.values()]))
                if value<best:best=value;best_epoch=epoch;save_predictor(p,out/'best.pt',epoch=epoch,dev_score=best)
                row=dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,mean_centered_cv_ratio=value,states=metrics,seconds=clock()-started);history.append(row);write(out/'history.json',history);log(dict(name=job['name'],epoch=epoch,best=best,seconds=row['seconds']))
        write(out/'summary.json',dict(complete=True,best_epoch=best_epoch,best_centered_cv_ratio=best,seconds=clock()-started,best_sha256=sha(out/'best.pt'),cache_preparation_seconds=manifest['total_seconds'],teacher_collection_and_trajectory_additional=True))
        del p,opt,states;gc.collect()

if __name__=='__main__':main()
