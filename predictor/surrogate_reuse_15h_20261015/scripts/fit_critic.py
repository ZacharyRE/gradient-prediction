"""Value/derivative supervision, with state-wise held-out gradient selection."""
from predictor_core import *
from loss_critic import LossCritic
from global_gradient_objective import estimate_factor_normalizers
from updates_legacy import clock
import argparse,gc

def evaluate_state(p,rows,A,B,s):
    pp=[];tt=[];values=[];actual=[]
    p.eval()
    for ids in sample_chunks(rows,max_n=16,budget=4096):
        d=pack([rows[i] for i in ids])
        condition_lora_state(p,d,A,B,s)
        with torch.no_grad():
            if isinstance(p,LossCritic):
                v,g=p.values_and_grad(d,False);values.extend(v.detach().cpu().tolist());actual.extend(rows[i]['loss_sum'] for i in ids)
            else:g=p(d)
            pg=raw(d['x'],g,A,B,s);tg=raw(d['x'],d['g'],A,B,s)
            pp.append(torch.cat([v.flatten(1) for v in pg],1));tt.append(torch.cat([v.flatten(1) for v in tg],1))
    pp=torch.cat(pp);tt=torch.cat(tt);pc=pp-pp.mean(0);tc=tt-tt.mean(0)
    result=dict(centered_cv_ratio=float((tc-pc).square().sum()/tc.square().sum().clamp_min(1e-20)),relative_l2=float((pp-tt).norm()/tt.norm().clamp_min(1e-20)),centered_correlation=float((pc*tc).sum()/(pc.norm()*tc.norm()).clamp_min(1e-20)))
    mean_error=(pp.mean(0)-tt.mean(0)).square().sum();residual_variance=(pc-tc).square().sum()/len(pp);true_mean=tt.mean(0).square().sum();true_variance=tc.square().sum()/len(tt)
    result.update(batch32_error_ratio=float((mean_error+residual_variance/32)/(true_mean+true_variance/32).clamp_min(1e-20)),mean_gradient_cosine=float(F.cosine_similarity(pp.mean(0),tt.mean(0),dim=0)),mean_gradient_squared_error=float(mean_error))
    if values:result['loss_relative_mse']=float(np.mean((np.array(values)-actual)**2)/max(1e-12,np.mean(np.array(actual)**2)))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--cache',default='critic_v1');a=ap.parse_args()
    from optional_fit_budget import registered_fit_cutoff
    completion_budget=registered_fit_cutoff(a.jobs,a.cache)
    gpu_guard();guard_space();seed(901);jobs=json.loads(Path(a.jobs).read_text());mean_preparation_seconds=0.
    source=EXP/'cache'/a.cache;manifest=json.loads((source/'complete.json').read_text());assert manifest['complete']
    states=[]
    for item in manifest['states']:
        name=item['state']['name'];path=source/name;meta=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False)
        train=load_samples(path/'predictor_train');dev=load_samples(path/'predictor_dev');A=meta['A'].cuda();B=meta['B'].cuda();s=meta['scale']
        normalizers=estimate_factor_normalizers(train,A,B,s)
        if any(j.get('mean_weight',0) for j in jobs):
            tick=clock();sums=[torch.zeros_like(A),torch.zeros_like(B)]
            with torch.no_grad():
                for ix in sample_chunks(train,max_n=16,budget=4096):
                    d=pack([train[i] for i in ix]);pair=raw(d['x'],d['g'],A,B,s)
                    for total,v in zip(sums,pair):total.add_(v.sum(0))
            normalizers['mean_energy']=[float((v/len(train)).square().sum()) for v in sums];mean_preparation_seconds+=clock()-tick
        states.append(dict(name=name,train=train,dev=dev,A=A,B=B,s=s,normalizers=normalizers))
    allrows=[r for st in states for r in st['train']];tokens=sorted({int(t) for r in allrows for t in r['label_ids'].tolist() if t>=0})
    target_scale=(sum(float(r['g'].square().sum()) for r in allrows)/sum(r['g'].numel() for r in allrows))**.5
    value_scale=max(1e-12,float(np.mean([r['loss_sum']**2 for r in allrows])))
    reserve_worker_cache()
    for job in jobs:
        out=EXP/'models'/f'mnli_critic_{job["name"]}'
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True);seed(job.get('seed',901));tick=clock()
        if job['kind']=='copied_suffix':
            from copied_suffix import CopiedSuffix
            from transformers import AutoConfig
            config=AutoConfig.from_pretrained(MODELS['1p5b'],local_files_only=True).to_dict()
            p=CopiedSuffix(states[0]['B'].shape[0],tokens,config,source_layers=job.get('source_layers',[18,27])).cuda();p.initialize_readout(MODELS['1p5b'])
        elif job['kind']=='token':
            from token_critic import TokenCritic
            p=TokenCritic(states[0]['B'].shape[0],width=job.get('width',128),depth=job.get('depth',2),label_tokens=tokens,pretrained_readout=job.get('pretrained_readout',True)).cuda()
            p.initialize_readout(MODELS['1p5b'])
        elif job['kind']=='critic':
            p=LossCritic(states[0]['B'].shape[0],width=job.get('width',128),depth=job.get('depth',2),label_tokens=tokens).cuda()
            p.loss_scale.fill_(max(.01,float(np.mean([r['loss_sum']/r['n'] for r in allrows]))))
        else:
            constructor=Predictor
            if job['kind']=='state':
                from state_predictor import StatePredictor
                constructor=StatePredictor
            p=constructor(states[0]['B'].shape[0],width=job.get('width',128),depth=job.get('depth',2),label_dim=len(tokens),label_vocab=max(tokens)+1,input_field='h').cuda()
            with torch.no_grad():p.label_table[torch.tensor(tokens,device='cuda'),torch.arange(len(tokens),device='cuda')]=1
        p.target_scale.fill_(target_scale);opt=torch.optim.AdamW(p.parameters(),lr=job.get('lr',.0003),weight_decay=.01)
        rng=np.random.default_rng(job.get('seed',901));history=[];best=float('inf');selected=None;optimizer_steps=0
        selection_metric=job.get('selection_metric','centered_cv_ratio');assert selection_metric in ['centered_cv_ratio','batch32_error_ratio']
        write(out/'config.json',dict(job=job,cache=str(source),cache_sha256=sha(source/'complete.json'),train_examples=len(allrows),states=[st['name'] for st in states],source_sha256=source_signature(__file__),selection=f'Mean within-state {selection_metric} on predictor_dev; no downstream final examples',batch32_metric_scope='Expected squared mean-gradient error for32 IID draws from the empirical validation-gradient distribution, including mean bias; a diagnostic proxy, not a downstream guarantee',completion_budget=completion_budget))
        for epoch in range(job.get('epochs',30)+1):
            if completion_budget is None:
                if deadline(7200):raise RuntimeError('Reserve final validation')
            elif time.time()>=completion_budget['cutoff_epoch']:
                raise RuntimeError('Registered post-validation fit completion cutoff reached')
            losses=[]
            if epoch:
                batches=[]
                for i,st in enumerate(states):
                    ids=list(map(int,rng.permutation(len(st['train']))));batches.extend((i,ix) for ix in sample_chunks(st['train'],ids,max_n=job.get('micro',16),budget=job.get('token_budget',4096)))
                rng.shuffle(batches)
                for i,ix in batches:
                    st=states[i];rr=[st['train'][j] for j in ix];d=pack(rr);A,B,s=st['A'],st['B'],st['s'];norm=st['normalizers'];opt.zero_grad(set_to_none=True);p.train()
                    condition_lora_state(p,d,A,B,s)
                    value_loss=torch.zeros((),device='cuda')
                    if isinstance(p,LossCritic):
                        if job.get('gradient_weight',1)>0:values,g=p.values_and_grad(d,True)
                        elif job.get('value_weight',1)>0:values=p.loss_values(d,d['h'].float());g=None
                        else:values=None;g=None
                        if values is not None:
                            target=torch.tensor([r['loss_sum'] for r in rr],device='cuda');value_loss=(values-target).square().mean()/value_scale
                    else:g=p(d)
                    gradient_loss=torch.zeros((),device='cuda')
                    if g is not None:
                        pa,pb=raw(d['x'],g,A,B,s)
                        with torch.no_grad():ta,tb=raw(d['x'],d['g'],A,B,s)
                        gradient_loss=.5*((pa-ta).square().sum((1,2)).mean()/norm['factor_A']+(pb-tb).square().sum((1,2)).mean()/norm['factor_B'])
                        gradient_loss=gradient_loss+.1*(g-d['g']).square().sum()/d['valid'].sum()/g.shape[-1]/p.target_scale.square()
                        if job.get('mean_weight',0):
                            mean_loss=0.
                            for j,(prediction,truth,key) in enumerate([(pa,ta,'factor_A'),(pb,tb,'factor_B')]):
                                expected_batch_energy=norm['mean_energy'][j]+max(0.,norm[key]-norm['mean_energy'][j])/len(rr)
                                mean_loss=mean_loss+.5*(prediction.mean(0)-truth.mean(0)).square().sum()/max(1e-12,expected_batch_energy)
                            gradient_loss=gradient_loss+job['mean_weight']*mean_loss
                    distillation_loss=torch.zeros((),device='cuda')
                    if job.get('distillation_weight',0):
                        assert job['kind'] in ['token','copied_suffix'] and p.pretrained_readout
                        teacher_hidden=torch.cat([r['teacher_hidden'] for r in rr]).to('cuda').float()
                        with torch.no_grad():teacher_logits=teacher_hidden@p.readout.T
                        student_logits=p.token_logits(d,d['h'].float())[d['active']]
                        assert teacher_logits.shape==student_logits.shape
                        temperature=job.get('distillation_temperature',2.)
                        distillation_loss=F.kl_div(F.log_softmax(student_logits/temperature,-1),F.softmax(teacher_logits/temperature,-1),reduction='batchmean')*temperature**2
                    loss=job.get('gradient_weight',1)*gradient_loss+job.get('value_weight',1)*value_loss+job.get('distillation_weight',0)*distillation_loss
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic training loss')
                    loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step();optimizer_steps+=1;losses.append(float(loss.detach()))
            metrics={st['name']:evaluate_state(p,st['dev'],st['A'],st['B'],st['s']) for st in states}
            score=float(np.mean([v[selection_metric] for v in metrics.values()]));centered_score=float(np.mean([v['centered_cv_ratio'] for v in metrics.values()]))
            if score<best:best=score;selected=epoch;selected_centered=centered_score;save_predictor(p,out/'best.pt',epoch=epoch,dev_score=score)
            row=dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,mean_centered_cv_ratio=centered_score,selection_metric=selection_metric,selection_score=score,states=metrics,seconds=clock()-tick);history.append(row);write(out/'history.json',history)
            log(dict(name=job['name'],epoch=epoch,best=best,selected_epoch=selected,seconds=row['seconds']))
        write(out/'summary.json',dict(complete=True,best_epoch=selected,best_centered_cv_ratio=selected_centered,selection_metric=selection_metric,best_selection_score=best,seconds=clock()-tick,mean_normalization_preparation_seconds=mean_preparation_seconds,optimizer_steps=optimizer_steps,best_sha256=sha(out/'best.pt'),value_and_gradient_supervision=job,collection_cost_seconds=sum(a['seconds'] for item in manifest['states'] for a in item['audits']),teacher_trajectory_cost_additional=True))
        del p,opt;gc.collect()

if __name__=='__main__':main()
