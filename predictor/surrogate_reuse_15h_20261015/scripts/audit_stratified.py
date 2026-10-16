"""Exhaustive finite-population unbiasedness check for stratified CV."""
from stratified import choose
from common import EXP,write
import numpy as np,itertools

rng=np.random.default_rng(42);groups={'a':[0,1,2],'b':[3,4],'c':[5]}
pos,w=choose(groups,4,rng)
counts={g:sum(i in ids for i in pos) for g,ids in groups.items()}
truth=rng.normal(size=(6,7));pred=rng.normal(size=(6,7));results=[]
for c in [0.,.5,1.,3.]:
    estimates=[]
    choices=[list(itertools.combinations(ids,counts[g])) for g,ids in groups.items()]
    for sets in itertools.product(*choices):
        estimate=c*pred.sum(0)
        for (g,ids),chosen in zip(groups.items(),sets):
            estimate=estimate+len(ids)/len(chosen)*(truth[list(chosen)]-c*pred[list(chosen)]).sum(0)
        estimates.append(estimate)
    bias=float(abs(np.mean(estimates,axis=0)-truth.sum(0)).max())
    assert bias<1e-12
    results.append(dict(coefficient=c,max_absolute_expectation_bias=bias))
write(EXP/'results/stratified_estimator_audit.json',dict(passed=True,checks=results,note='Estimator algebra; raw-gradient reconstruction precision is covered by separate numerical audit.'))
print(results)
