"""Fit teacher class-logit surrogate, with explicit within-example state changes."""
from predictor_core import *
from behavior_surrogate import BehaviorSurrogate
from token_critic import TokenCritic
from updates_legacy import clock
import argparse,gc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);a=ap.parse_args();gpu_guard();guard_space();seed(991)
    source=EXP/'cache/critic_hidden';manifest=json.loads((source/'complete.json').read_text());started=clock()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODELS['1p5b'],local_files_only=True);tokens=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABC']
    readout=TokenCritic(1536,width=16,depth=1,label_tokens=tokens).cuda();readout.initialize_readout(MODELS['1p5b']);head=readout.readout.clone();del readout
    prototype=BehaviorSurrogate(class_tokens=tokens).cuda();states=[]
    for item in manifest['states']:
        path=source/item['state']['name'];meta=torch.load(path/'metadata.pt',map_location='cpu',weights_only=False);state=dict(name=item['state']['name'])
        A,B,s=meta['A'].cuda(),meta['B'].cuda(),meta['scale']
        for split in ['predictor_train','predictor_dev']:
            rows=load_samples(path/split);parts={k:[] for k in ['features','logits','classes']}
            for ix in sample_chunks(rows,max_n=16,budget=4096):
                rr=[rows[i] for i in ix];d=pack(rr)
                with torch.no_grad():features=prototype.features(d,A,B,s)
                hidden=[];classes=[]
                for r in rr:
                    target_ids=r['label_ids'][r['active']];mask=(target_ids[:,None]==torch.tensor(tokens)[None,:]).any(-1)
                    assert int(mask.sum())==1
                    hidden.append(r['teacher_hidden'][mask]);classes.append(tokens.index(int(target_ids[mask][0])))
                logits=torch.cat(hidden).cuda().float()@head.T;logits-=logits.mean(-1,keepdim=True)
                parts['features'].append(features);parts['logits'].append(logits);parts['classes'].append(torch.tensor(classes,device='cuda'))
            state[split]={k:torch.cat(v) for k,v in parts.items()};state[split]['indices']=[r['index'] for r in rows]
        states.append(state)
    del prototype;prep=clock()-started
    for split in ['predictor_train','predictor_dev']:
        assert all(st[split]['indices']==states[0][split]['indices'] for st in states),'State contrasts require paired examples'
    cache={split:{k:torch.stack([st[split][k] for st in states]) for k in ['features','logits','classes']} for split in ['predictor_train','predictor_dev']}
    reserve_worker_cache()
    for job in json.loads(Path(a.jobs).read_text()):
        out=EXP/'models'/f'mnli_behavior_{job["name"]}'
        if (out/'summary.json').exists():continue
        out.mkdir(exist_ok=True);seed(991);tick=clock();p=BehaviorSurrogate(width=job.get('width',256),class_tokens=tokens,delta_features=job.get('delta_features',True)).cuda()
        train=cache['predictor_train'];dev=cache['predictor_dev'];fd=p.feature_mean.numel();x=train['features'][...,:fd];y=train['logits'];dx=dev['features'][...,:fd];dy=dev['logits']
        with torch.no_grad():p.feature_mean.copy_(x.mean((0,1)));scale=x.std((0,1));p.feature_scale.copy_(scale.clamp_min(.01))
        normalizer=y.square().mean().clamp_min(1e-8);change_scale=(y-y.mean(0,keepdim=True)).square().mean().clamp_min(1e-8)
        opt=torch.optim.AdamW(p.parameters(),lr=job.get('lr',.0003),weight_decay=.01);rng=np.random.default_rng(991);best=float('inf');history=[]
        for epoch in range(job.get('epochs',200)+1):
            if deadline(7200):raise RuntimeError('Final validation reserve')
            if epoch:
                for ids in np.array_split(rng.permutation(x.shape[1]),max(1,x.shape[1]//64)):
                    pred=p(x[:,ids]);truth=y[:,ids];loss=(pred-truth).square().mean()/normalizer
                    loss+=job.get('change_weight',1.)*((pred-pred.mean(0,keepdim=True))-(truth-truth.mean(0,keepdim=True))).square().mean()/change_scale
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step()
            if epoch%5==0:
                with torch.no_grad():
                    pred=p(dx);mse=float((pred-dy).square().mean()/normalizer);delta_mse=float(((pred-pred.mean(0,keepdim=True))-(dy-dy.mean(0,keepdim=True))).square().mean()/change_scale);agreement=float((pred.argmax(-1)==dy.argmax(-1)).float().mean())
                    perstate=[dict(name=st['name'],student_accuracy=float((pred[i].argmax(-1)==dev['classes'][i]).float().mean()),teacher_forced_accuracy=float((dy[i].argmax(-1)==dev['classes'][i]).float().mean())) for i,st in enumerate(states)]
                score=mse+job.get('change_weight',1.)*delta_mse
                if score<best:best=score;selected=epoch;torch.save(dict(config=p.cfg,state_dict=p.state_dict(),epoch=epoch),out/'best.pt')
                row=dict(epoch=epoch,normalized_logit_mse=mse,normalized_state_change_mse=delta_mse,teacher_class_agreement=agreement,states=perstate,seconds=clock()-tick);history.append(row);write(out/'history.json',history)
        write(out/'config.json',dict(job=job,source_cache_sha256=sha(source/'complete.json'),source_sha256=source_signature(__file__),selection='predictor_dev normalized class-logit error and paired within-example state-change error',no_gold_class_features=True))
        write(out/'summary.json',dict(complete=True,best_epoch=selected,best_score=best,fit_seconds=clock()-tick,feature_preparation_seconds=prep,best_sha256=sha(out/'best.pt'),teacher_collection_and_trajectories_additional=True));log(dict(name=job['name'],best_epoch=selected,best_score=best,seconds=clock()-tick));del p,opt;gc.collect()

if __name__=='__main__':main()
