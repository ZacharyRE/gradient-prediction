"""Extend cache numerical checks to newly available 512-update full-prefix runs."""
from common import *
from safetensors.numpy import load_file


def main():
    rows=[]
    for task in ['paws','mnli']:
        for method in ['oracle32','oracle8','oracle16','cv8','cv16']:
            for lr in [.0003,.001]:
                cached=EXP/'models'/f'{task}__tcache_{method}_lr{lr}_u101_p123/step-512/adapter_model.safetensors'
                reference=EXP/'models'/f'{task}__long_{method}_lr{lr}_u101_p123/step-512/adapter_model.safetensors'
                assert cached.exists() and reference.exists()
                a,b=load_file(str(cached)),load_file(str(reference));assert a.keys()==b.keys()
                numerator=sum(float(np.square(a[k].astype('float64')-b[k].astype('float64')).sum()) for k in a)
                denominator=sum(float(np.square(b[k].astype('float64')).sum()) for k in b)
                rows.append(dict(task=task,method=method,lr=lr,steps=512,seed=101,bitwise_equal=all(np.array_equal(a[k],b[k]) for k in a),relative_weight_difference=(numerator/max(denominator,1e-30))**.5,cached_sha256=sha(cached),reference_sha256=sha(reference)))
    write(EXP/'results/teacher_cache_long_comparison.json',dict(rows=rows,scope='Seed101 saved weights at512updates: research_teacher_cache versus research_v3 long trajectories. A numerical audit of existing artifacts, not a new training run. Earlier equality at256 does not guarantee later equality. Independently evaluate non-identical adapters; never attach their timings to another adapter\'s accuracy.'))
    log(dict(comparisons=len(rows),bitwise_equal=sum(r['bitwise_equal'] for r in rows),different=[r for r in rows if not r['bitwise_equal']]))


if __name__=='__main__':main()
