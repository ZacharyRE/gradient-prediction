"""Fit behavior changes, retaining a paid initial-teacher readout exactly."""
from predictor_core import *
from behavior_surrogate import BehaviorSurrogate,AnchoredBehaviorSurrogate
from token_critic import TokenCritic
from updates_legacy import clock
import argparse,gc

def training_features(cache_name='critic_hidden'):
    source=EXP/'cache'/cache_name;manifest=json.loads((source/'complete.json').read_text());started=clock()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS['1p5b'],local_files_only=True);tokens=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABC']
    readout=TokenCritic(1536,width=16,depth=1,label_tokens=tokens).cuda();readout.initialize_readout(MODELS['1p5b']);head=readout.readout.clone();del readout
    prototype=BehaviorSurrogate(class_tokens=tokens).cuda();states=[]
    for item in manifest['states']:
        path=source/item['state']['name'];meta=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False);state=dict(item['state'])
        A,B,s=meta['A'].cuda(),meta['B'].cuda(),meta['scale']
        for split in ['predictor_train','predictor_dev']:
            rows=load_samples(path/split);parts={k:[] for k in ['features','logits','classes']}
            for ix in sample_chunks(rows,max_n=16,budget=4096):
                rr=[rows[i] for i in ix];d=pack(rr)
                with torch.no_grad():features=prototype.features(d,A,B,s)
                hidden=[];classes=[]
                for r in rr:
                    ids=r['label_ids'][r['active']];mask=(ids[:,None]==torch.tensor(tokens)[None,:]).any(-1);assert int(mask.sum())==1
                    hidden.append(r['teacher_hidden'][mask]);classes.append(tokens.index(int(ids[mask][0])))
                logits=torch.cat(hidden).cuda().float()@head.T;logits-=logits.mean(-1,keepdim=True)
                parts['features'].append(features);parts['logits'].append(logits);parts['classes'].append(torch.tensor(classes,device='cuda'))
            state[split]={k:torch.cat(v) for k,v in parts.items()};state[split]['indices']=[r['index'] for r in rows]
        states.append(state)
    for split in ['predictor_train','predictor_dev']:assert all(st[split]['indices']==states[0][split]['indices'] for st in states)
    anchors={st['warmup_seed']:i for i,st in enumerate(states) if st['step']==0};indices=torch.tensor([anchors[st['warmup_seed']] for st in states],device='cuda')
    cache={split:{k:torch.stack([st[split][k] for st in states]) for k in ['features','logits','classes']} for split in ['predictor_train','predictor_dev']}
    return cache,indices,tokens,clock()-started,source,states

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--cache',default='critic_hidden');a=ap.parse_args();gpu_guard();guard_space();seed(991)
    cache,anchor_indices,tokens,prep,source,states=training_features(a.cache);reserve_worker_cache()
    train=cache['predictor_train'];dev=cache['predictor_dev'];x=train['features'];y=train['logits'];dx=dev['features'];dy=dev['logits'];initial=x[anchor_indices];initial_y=y[anchor_indices];di=dx[anchor_indices];diy=dy[anchor_indices]
    normalizer=y.square().mean().clamp_min(1e-8);change_scale=(y-y.mean(0,keepdim=True)).square().mean().clamp_min(1e-8)
    def metrics(pred):
        mse=float((pred-dy).square().mean()/normalizer);change=float(((pred-pred.mean(0,keepdim=True))-(dy-dy.mean(0,keepdim=True))).square().mean()/change_scale)
        return dict(normalized_logit_mse=mse,normalized_state_change_mse=change,joint_score=mse+change,teacher_class_agreement=float((pred.argmax(-1)==dy.argmax(-1)).float().mean()),student_accuracy=float((pred.argmax(-1)==dev['classes']).float().mean()))
    baseline=metrics(diy)
    for job in json.loads(Path(a.jobs).read_text()):
        out=EXP/'models'/f'mnli_behavior_{job["name"]}'
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True);seed(991);tick=clock();p=AnchoredBehaviorSurrogate(width=job.get('width',128),class_tokens=tokens).cuda()
        with torch.no_grad():z=p.inputs(x,initial,initial_y);p.feature_mean.copy_(z.mean((0,1)));p.feature_scale.copy_(z.std((0,1)).clamp_min(.01));del z
        opt=torch.optim.AdamW(p.parameters(),lr=job.get('lr',.0001),weight_decay=job.get('weight_decay',.01));rng=np.random.default_rng(991);best=float('inf');history=[]
        for epoch in range(job.get('epochs',200)+1):
            if deadline(7200):raise RuntimeError('Final validation reserve')
            if epoch:
                for ids in np.array_split(rng.permutation(x.shape[1]),max(1,x.shape[1]//64)):
                    pred=p(x[:,ids],initial[:,ids],initial_y[:,ids]);truth=y[:,ids];loss=(pred-truth).square().mean()/normalizer
                    loss+=((pred-pred.mean(0,keepdim=True))-(truth-truth.mean(0,keepdim=True))).square().mean()/change_scale
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step()
            if epoch%5==0:
                with torch.no_grad():pred=p(dx,di,diy);row=metrics(pred);at_anchor=p(di,di,diy);assert torch.equal(at_anchor,diy),'Initial agent must be preserved exactly'
                if row['joint_score']<best:best=row['joint_score'];selected=epoch;torch.save(dict(config=p.cfg,state_dict=p.state_dict(),epoch=epoch),out/'best.pt')
                row.update(epoch=epoch,seconds=clock()-tick);history.append(row);write(out/'history.json',history)
        write(out/'config.json',dict(job=job,cache=str(source),source_cache_sha256=sha(source/'complete.json'),source_sha256=source_signature(__file__),selection='Predictor_dev normalized full-logit plus within-example state-change MSE; compare models across differing state coverage only after common-state rescoring',initial_teacher_readout_required=True,no_gold_class_features=True,states=[{k:st[k] for k in ['name','warmup_seed','step']} for st in states]))
        write(out/'summary.json',dict(complete=True,best_epoch=selected,best_score=best,fit_seconds=clock()-tick,feature_preparation_seconds=prep,best_sha256=sha(out/'best.pt'),initial_teacher_baseline=baseline,teacher_collection_and_trajectories_additional=True));log(dict(name=job['name'],best_epoch=selected,best_score=best,baseline=baseline));del p,opt;gc.collect()

if __name__=='__main__':main()
