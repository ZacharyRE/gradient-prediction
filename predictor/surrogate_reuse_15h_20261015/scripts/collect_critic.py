"""Collect scalar teacher losses and activation derivatives on training-only rows."""
from predictor_core import *
from fast_backend import configure
from updates_legacy import clock
from safetensors.torch import load_file
import argparse,gc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--states',required=True);ap.add_argument('--name',default='critic_v1');ap.add_argument('--train-examples',type=int,default=512);ap.add_argument('--include-hidden',action='store_true');a=ap.parse_args()
    states=json.loads(Path(a.states).read_text());out=EXP/'cache'/a.name;out.mkdir(exist_ok=True)
    m,tok,mod=setup('1p5b',states[0]['adapter'],rng=777);cap=Capture(mod,m);A,B,s=weights(mod);configure(True,8192)
    datasets={split:Dataset('mnli',split,tok) for split in ['predictor_train','predictor_dev']};receipts=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    for state in states:
        if deadline(7200):raise RuntimeError('Reserve validation time')
        dest=out/state['name'];dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():receipts.append(json.loads((dest/'complete.json').read_text()));continue
        vals=load_file(str(Path(state['adapter'])/'adapter_model.safetensors'))
        with torch.no_grad():
            A.copy_(next(v for k,v in vals.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in vals.items() if k.endswith('lora_B.weight')))
        torch.save(dict(A=A.detach().cpu(),B=B.detach().cpu(),scale=s,state=state),dest/'metadata.pt')
        audits=[]
        for split,ds in datasets.items():
            sd=dest/split;sd.mkdir(exist_ok=True)
            ids=list(map(int,np.random.default_rng(314159).permutation(len(ds))[:a.train_examples])) if split=='predictor_train' else list(range(len(ds)))
            tick=clock();shards=[];grad_audits=[]
            for shard_id,ix in enumerate(chunks(ds,ids,max_n=16,token_budget=8192)):
                bb=batch(ds,ix,tok);m.zero_grad(set_to_none=True);N=sum(ds[i]['n'] for i in ix)
                with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.MATH):
                    base=m.get_base_model();hidden=base.model(input_ids=bb['input_ids'],attention_mask=bb['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
                    labels=bb['labels'][:,1:];valid=labels!=-100;logits=base.lm_head(hidden[:,:-1][valid])
                    mean_loss=F.cross_entropy(logits.float(),labels[valid],reduction='mean')
                    token_losses=F.cross_entropy(logits.detach().float(),labels[valid],reduction='none')
                (mean_loss*N).backward()
                assert torch.isfinite(mean_loss) and torch.isfinite(cap.y.grad).all()
                with torch.no_grad():
                    ra,rb=raw(cap.x.detach(),cap.y.grad.detach(),A,B,s)
                    grad_audits.append(measures(flat((ra.sum(0),rb.sum(0))),flat((A.grad,B.grad))))
                losses=torch.zeros_like(labels,dtype=torch.float32);losses[valid]=token_losses;per=losses.sum(1)
                rows=[]
                for j,i in enumerate(ix):
                    L=len(ds[i]['input_ids']);shifted=torch.tensor(ds[i]['labels'][1:]+[-100])
                    rows.append(dict(index=i,x=cap.x[j,:L].detach().cpu().bfloat16(),y=cap.y[j,:L].detach().cpu().bfloat16(),h=cap.h[j,:L].detach().cpu().bfloat16(),g=cap.y.grad[j,:L].detach().cpu().float(),active=shifted>=0,label_ids=shifted,n=ds[i]['n'],loss_sum=float(per[j]),sample_id=ds.rows[i]['sample_id']))
                    if a.include_hidden:rows[-1]['teacher_hidden']=hidden[j,:L][(shifted>=0).to(hidden.device)].detach().cpu().bfloat16()
                path=sd/f'shard{shard_id:04d}.pt';torch.save(rows,path);shards.append({'name':path.name,'n':len(rows),'sha256':sha(path)})
                del hidden,logits,mean_loss,token_losses,losses,rows,bb,ra,rb;m.zero_grad(set_to_none=True)
            receipt=dict(split=split,n=len(ids),seconds=clock()-tick,indices=ids,data_sha256=sha(data('mnli',split)),shards=shards,max_reconstruction_relative_l2=max(r['relative_l2'] for r in grad_audits),full_inputs=True)
            write(sd/'manifest.json',receipt);audits.append(receipt)
        rec=dict(state=state,adapter_sha256=sha(Path(state['adapter'])/'adapter_model.safetensors'),audits=audits,complete=True)
        write(dest/'complete.json',rec);receipts.append(rec);log(dict(state=state['name'],seconds=sum(x['seconds'] for x in audits)))
    cap.close();write(out/'complete.json',dict(complete=True,states=receipts,source_sha256=source_signature(__file__),purpose='Loss and gradient fitting; original predictor_train/dev only; no selection or final evaluation inputs'))

if __name__=='__main__':main()
