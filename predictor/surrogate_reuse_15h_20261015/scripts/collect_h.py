"""Fresh consistent residual-stream features and true gradients on fit-only splits."""
from research_h import *
from fast_backend import configure

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);a=ap.parse_args()
    task=a.task;out=EXP/'cache'/f'{task}_h'
    if (out/'complete.json').exists():return
    out.mkdir(exist_ok=True);warm=json.loads((EXP/'configs'/f'{task}_warmup.json').read_text())
    m,tok,mod=setup('1p5b',warm['adapter'],rng=201);cap=Capture(mod,m);A,B,s=weights(mod);configure(True,8192)
    torch.save(dict(A=A.detach().cpu(),B=B.detach().cpu(),scale=s),out/'metadata.pt');audits=[]
    for split in ['predictor_train','predictor_dev']:
        dest=out/split
        if (dest/'manifest.json').exists():audits.append(json.loads((dest/'manifest.json').read_text()));continue
        dest.mkdir(exist_ok=True);ds=Dataset(task,split,tok);tick=clock();checks=[]
        for st in range(0,len(ds),32):
            rows,_,check=collect(m,tok,mod,cap,ds,list(range(st,min(st+32,len(ds)))),micro=32,keep=True,verify_reconstruction=True)
            assert all('h' in r for r in rows)
            assert check['min_reconstruction_cosine']>.999 and check['max_reconstruction_relative_l2']<.02
            torch.save(rows,dest/f'shard{st//32:04d}.pt');checks.append(check)
        record=dict(split=split,n=len(ds),seconds=clock()-tick,full_inputs=True,source_sha256=sha(data(task,split)),min_reconstruction_cosine=min(c['min_reconstruction_cosine'] for c in checks),max_reconstruction_relative_l2=max(c['max_reconstruction_relative_l2'] for c in checks))
        write(dest/'manifest.json',record);audits.append(record);log(record)
    write(out/'complete.json',dict(complete=True,audits=audits,start_sha256=warm['adapter_sha256'],backend='supervised-only logits, micro32, full transformer',code_sha256=sha(Path(__file__))))
    cap.close()

if __name__=='__main__':main()
