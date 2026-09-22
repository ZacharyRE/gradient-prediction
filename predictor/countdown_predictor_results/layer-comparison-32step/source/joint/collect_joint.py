"""Cache full-sequence gradients at the jointly warmed three-layer state."""
from three_layers import *

def main():
    m,tok,mods=setup();m.eval();cap=Captures(mods);start=time.time();audits={l:[] for l in mods};sizes={}
    for l in mods:(EXP/'cache'/f'layer{l:02d}').mkdir(exist_ok=False)
    for split in ['predictor_train','predictor_extra','predictor_dev']:
        ds=CompletionDataset(data_path(split),tok,m.config.max_position_embeddings)
        sizes[split]=dict(n=len(ds),lengths=ds.length_statistics)
        for l in mods:(EXP/'cache'/f'layer{l:02d}'/split).mkdir()
        shards={l:[] for l in mods};sn=0
        for st in range(0,len(ds),8):
            ids=list(range(st,min(st+8,len(ds))));bb=batch(ds,ids,tok);m.zero_grad(set_to_none=True)
            n=int((bb['labels'][:,1:]!=-100).sum());(forward(m,bb).loss*n).backward()
            for l,mod in mods.items():
                A,B,scale=weights(mod);x=cap.x[l].float();y=cap.y[l].detach().float();g=cap.y[l].grad.float()
                if st==0:
                    ga,gb=raw_grads(x.flatten(0,1),g.flatten(0,1),A,B,scale)
                    audit=dict(split=split,A=measures(ga,A.grad),B=measures(gb,B.grad))
                    assert min(audit[k]['cosine'] for k in ['A','B'])>.999,audit
                    audits[l].append(audit)
                for j,i in enumerate(ids):
                    length=len(ds[i]['input_ids']);target=torch.tensor(ds[i]['labels'][1:]+[-100])
                    shards[l].append(dict(index=i,x=x[j,:length].cpu().bfloat16(),y=y[j,:length].cpu().bfloat16(),g=g[j,:length].detach().cpu(),active=target>=0,n=int((target>=0).sum())))
            if len(shards[next(iter(mods))])>=64 or st+8>=len(ds):
                for l in mods:torch.save(shards[l],EXP/'cache'/f'layer{l:02d}'/split/f'shard{sn:04d}.pt');shards[l]=[]
                sn+=1
            if st%256==0:print(json.dumps(dict(split=split,seen=st+len(ids),seconds=time.time()-start)),flush=True)
    for l,mod in mods.items():
        A,B,scale=weights(mod);out=EXP/'cache'/f'layer{l:02d}'
        torch.save(dict(A=A.detach().cpu(),B=B.detach().cpu(),scale=scale),out/'metadata.pt')
        write(out/'manifest.json',dict(complete=True,layer=l,adapter=str(EXP/'models/teacher/step-32'),target='d(sum supervised next-token CE)/d(o_proj direct output)',splits=sizes,audits=audits[l],input_truncation=False,elapsed=time.time()-start))
    cap.close()
    write(EXP/'results/collection.json',dict(complete=True,layers=list(mods),splits=sizes,elapsed=time.time()-start))
if __name__=='__main__':main()
