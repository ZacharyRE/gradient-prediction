"""Compare completed cache/full-prefix trajectories on CPU, with artifact receipts."""
from common import *
import torch
from safetensors.torch import load_file

def main():
    torch.set_num_threads(4);rows=[]
    for task in ['paws','mnli']:
        for method in ['oracle32','oracle8','oracle16','cv8','cv16']:
            for lr in [.0003,.001]:
                current=EXP/'models'/f'{task}__tcache_{method}_lr{lr}_u101_p123'
                if not (current/'summary.json').exists():continue
                old='fast_'+method
                if lr==.0003 and method.startswith('cv'):old+='_cache'
                elif lr==.001:
                    old='gpu_cv16_cached_lr0.001' if method=='cv16' else old+'_lr0.001'
                reference=EXP/'models'/f'{task}__{old}_u101_p123'
                for step in [32,64,128,256]:
                    x=current/f'step-{step}'/'adapter_model.safetensors';y=reference/f'step-{step}'/'adapter_model.safetensors'
                    if not x.exists() or not y.exists():continue
                    a=load_file(str(x));b=load_file(str(y));assert a.keys()==b.keys()
                    maximum=max(float((a[k]-b[k]).abs().max()) for k in a)
                    relative=(sum(float((a[k].double()-b[k].double()).square().sum()) for k in a)/max(1e-30,sum(float(b[k].double().square().sum()) for k in a)))**.5
                    rows.append(dict(task=task,method=method,lr=lr,step=step,seed=101,bitwise_equal=all(torch.equal(a[k],b[k]) for k in a),max_absolute_difference=maximum,relative_difference=relative,cached_adapter=str(x.relative_to(EXP)),reference_adapter=str(y.relative_to(EXP)),cached_sha256=sha(x),reference_sha256=sha(y)))
    write(EXP/'results/teacher_cache_trajectory_comparison.json',dict(updated=time.time(),rows=rows,scope='Actual saved model weights on completed trajectories, compared on CPU. Different BF16 batch construction may in general change rounding; equality is an observed result for these artifacts, not a guarantee for all inputs.'))
    log(dict(comparisons=len(rows),bitwise_equal=sum(r['bitwise_equal'] for r in rows)))

if __name__=='__main__':main()
