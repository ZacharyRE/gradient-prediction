"""Fit-only examples at several Oracle states; never use evaluation labels to fit."""
from research_v3 import *
from fast_backend import configure
from safetensors.torch import load_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);a=ap.parse_args();task=a.task
    out=EXP/'cache'/f'{task}_multistate'
    if (out/'complete.json').exists():return
    out.mkdir(exist_ok=True);warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=201);cap=Capture(mod,m);A,B,s=weights(mod);a0=A.detach().clone();b0=B.detach().clone();configure(True,8192)
    states=[]
    for step in [0,32,128,256]:
        adapter=Path(warm['adapter']) if not step else EXP/'models'/f'{task}__fast_oracle32_u101_p123'/f'step-{step}'
        values=load_file(str(adapter/'adapter_model.safetensors'))
        with torch.no_grad():
            A.copy_(next(v for k,v in values.items() if k.endswith('lora_A.weight')));B.copy_(next(v for k,v in values.items() if k.endswith('lora_B.weight')))
        state=out/f'state{step}';state.mkdir(exist_ok=True)
        torch.save(dict(A=A.detach().cpu(),B=B.detach().cpu(),scale=s,warm_A=a0.cpu(),warm_B=b0.cpu()),state/'metadata.pt')
        audits=[]
        for split in ['predictor_train','predictor_dev']:
            dest=state/split
            if (dest/'manifest.json').exists():audits.append(json.loads((dest/'manifest.json').read_text()));continue
            dest.mkdir(exist_ok=True);ds=Dataset(task,split,tok)
            ids=list(map(int,np.random.default_rng(314159).permutation(len(ds))[:512])) if split=='predictor_train' else list(range(len(ds)))
            tick=clock()
            for st in range(0,len(ids),32):
                rows,_,_=collect(m,tok,mod,cap,ds,ids[st:st+32],micro=32,keep=True,verify_reconstruction=False)
                with torch.no_grad():
                    for r in rows:
                        x=r['x'].cuda().float()
                        r['delta_y']=(s*((x@A.float().T)@(B.float()-b0.float()).T+(x@(A.float()-a0.float()).T)@b0.float().T)).cpu()
                torch.save(rows,dest/f'shard{st//32:04d}.pt')
            audit=dict(split=split,n=len(ids),seconds=clock()-tick,source_sha256=sha(data(task,split)),selected_indices=ids,full_inputs=True)
            write(dest/'manifest.json',audit);audits.append(audit)
        record=dict(step=step,adapter=str(adapter),adapter_sha256=sha(adapter/'adapter_model.safetensors'),audits=audits)
        states.append(record);log(dict(task=task,state=step,collection_seconds=sum(r['seconds'] for r in audits)))
    write(out/'complete.json',dict(complete=True,states=states,selection='Fixed training split and predictor-development split; no downstream evaluation labels',teacher_trajectory='Oracle32 LR3e-4 seed101 from shared warmup; its256-update cost is additional preparation for a cold multi-state predictor',code_sha256=sha(Path(__file__))))
    cap.close()

if __name__=='__main__':main()
