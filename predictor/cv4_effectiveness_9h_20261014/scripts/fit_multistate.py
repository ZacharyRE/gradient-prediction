"""Train on several Oracle states, selecting only on predictor-development gradients."""
from research_v3 import *
import state_features as sf
sf.install()
pack=sf.state_pack
from global_gradient_objective import estimate_factor_normalizers,objective_global
from refit_predictor import score

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--predictor-seed',type=int,default=123);a=ap.parse_args()
    task=a.task;gpu_guard();seed(a.predictor_seed);reserve_worker_cache()
    cache=EXP/'cache'/f'{task}_multistate';registration=json.loads((cache/'complete.json').read_text());states=[]
    signature=source_signature(Path(__file__));snapshot=EXP/'provenance/source_versions'/signature;snapshot.mkdir(parents=True,exist_ok=True)
    for script in (EXP/'scripts').glob('*.py'):
        if not (snapshot/script.name).exists():shutil.copy2(script,snapshot/script.name)
    for state in registration['states']:
        path=cache/f'state{state["step"]}';meta=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False)
        train=load_samples(path/'predictor_train');dev=load_samples(path/'predictor_dev')
        A=meta['A'].cuda();B=meta['B'].cuda();s=meta['scale']
        states.append(dict(step=state['step'],A=A,B=B,s=s,train=train,dev=dev,normalizers=estimate_factor_normalizers(train,A,B,s)))
    for job in json.loads(Path(a.jobs).read_text()):
        out=EXP/'models'/f'{task}__multistate_{job["name"]}_p{a.predictor_seed}'
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True);seed(a.predictor_seed);tick=clock()
        base=EXP/'models'/f'{task}__refit_h_{job["base"]}_p{a.predictor_seed}'/'best.pt'
        p,_=load_predictor(base)
        if job.get('explicit_state'):
            z=states[0];p=sf.augment(p,z['A'],z['B'],z['s'],z['A'],z['B'],dict(delta_gain=job.get('delta_gain',100.),cal_scope='all'))
        opt=torch.optim.AdamW(p.parameters(),lr=job.get('fit_lr',.0003),weight_decay=.01);rng=np.random.default_rng(a.predictor_seed)
        history=[];best=float('inf');best_epoch=None
        write(out/'config.json',dict(task=task,job=job,predictor_seed=a.predictor_seed,base_predictor=str(base),base_predictor_sha256=sha(base),cache_sha256=sha(cache/'complete.json'),selection='Mean centered CV residual variance ratio across held-out predictor_dev examples at four fixed Oracle states',teacher_update_seed=101,independent_trajectory_validation_required=True,source_bundle_sha256=signature,code_sha256=sha(Path(__file__))))
        for epoch in range(job.get('epochs',20)+1):
            if deadline(5400):raise RuntimeError('Final validation reserve')
            losses=[]
            if epoch:
                batches=[]
                for st in states:
                    for ids in sample_chunks(st['train'],list(rng.permutation(len(st['train']))),max_n=32,budget=8192):batches.append((st,ids))
                rng.shuffle(batches)
                for st,ids in batches:
                    p.train();d=pack([st['train'][i] for i in ids]);opt.zero_grad(set_to_none=True)
                    loss=objective_global(p,d,st['A'],st['B'],st['s'],normalizers=st['normalizers'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite multi-state loss')
                    loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            p.eval();metrics=[]
            for st in states:
                result=score(p,st['dev'],st['A'],st['B'],st['s'])
                metrics.append(dict(step=st['step'],centered_cv_ratio=result['centered_cv_ratio'],centered_correlation=result['centered_correlation'],batch_lora=result['batch_lora']))
            value=float(np.mean([m['centered_cv_ratio'] for m in metrics]))
            if value<best:
                best=value;best_epoch=epoch;save_predictor(p,out/'best.pt',epoch=epoch,dev_score=value,explicit_state=job.get('explicit_state',False),delta_gain=job.get('delta_gain',100.))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,dev_states=metrics,mean_cv_ratio=value,seconds=clock()-tick));write(out/'history.json',history)
            if epoch%5==0:log(dict(task=task,multistate=job['name'],epoch=epoch,best=best,seconds=clock()-tick))
        write(out/'summary.json',dict(complete=True,best_epoch=best_epoch,best_centered_cv_ratio=best,seconds=clock()-tick,best_sha256=sha(out/'best.pt'),base_predictor=str(base),additional_collection_cost=sum(v['seconds'] for st in registration['states'] for v in st['audits']),teacher_trajectory_steps=256,teacher_trajectory_must_be_charged_for_cold_setup=True))
        del p,opt;gc.collect()

if __name__=='__main__':main()
