"""Fresh LoRA initializations and data orders; same declared warmup population."""
from predictor_core import *
from fast_backend import configure
from updates_legacy import clock
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',default='mnli');ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    out=EXP/'models'/f'{a.task}_warmup_w{a.seed}'
    assert not (out/'summary.json').exists()
    started=time.time();m,tok,mod=setup('1p5b',rng=a.seed);A,B,_=weights(mod);base=versions(m)
    ds=Dataset(a.task,'warmup',tok);configure(True,8192)
    opt=torch.optim.AdamW([A,B],lr=.001,weight_decay=0)
    rng=np.random.default_rng(a.seed);order=[]
    while len(order)<32*32:order.extend(map(int,rng.permutation(len(ds))))
    history=[]
    for step in range(1,33):
        if deadline(900):raise RuntimeError('Deadline')
        tick=clock();row=oracle_step(m,tok,mod,ds,order[(step-1)*32:step*32],opt,micro=32)
        history.append(dict(step=step,seconds=clock()-tick,**row))
    assert versions(m)==base
    m.save_pretrained(out)
    entry=dict(task=a.task,tag=f'{a.task}_warmup_w{a.seed}',adapter=str(out),method='warmup',warmup_seed=a.seed,seed=a.seed,steps=0,adapter_sha256=sha(out/'adapter_model.safetensors'))
    receipt=dict(entry=entry,warmup_steps=32,batch_size=32,lr=.001,data_sha256=sha(data(a.task,'warmup')),full_inputs=True,history=history,online_seconds=sum(r['seconds'] for r in history),wall_seconds=time.time()-started,source_sha256=source_signature(__file__))
    write(out/'summary.json',receipt);write(EXP/'configs'/f'{a.task}_warmup_w{a.seed}.json',entry);log(receipt)

if __name__=='__main__':main()
