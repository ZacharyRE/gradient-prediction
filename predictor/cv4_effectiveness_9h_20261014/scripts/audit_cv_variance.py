"""Check finite-population CV variance and quantify predictor accuracy targets."""
from common import *
import itertools

def main():
    rng=np.random.default_rng(4);n=8;truth=rng.normal(size=(n,5));pred=.4*truth+rng.normal(size=(n,5))*.2
    residual=truth-pred;s2=float(((residual-residual.mean(0))**2).sum()/(n-1));checks=[]
    for k in [1,2,4,8]:
        errors=[]
        for ids in itertools.combinations(range(n),k):
            estimator=pred.mean(0)+residual[list(ids)].mean(0)
            errors.append(float(((estimator-truth.mean(0))**2).sum()))
        observed=float(np.mean(errors));expected=(1/k-1/n)*s2
        assert abs(observed-expected)<1e-12
        checks.append(dict(n=n,k=k,observed=observed,formula=expected))
    targets=[]
    for k,m in [(4,8),(4,16),(8,16)]:
        ratio=(1/m-1/32)/(1/k-1/32)
        targets.append(dict(batch_size=32,cv_true_examples=k,oracle_true_examples=m,maximum_centered_residual_variance_ratio=ratio,required_variance_reduction_fraction=1-ratio))
    write(EXP/'results/cv_variance_audit.json',dict(passed=True,checks=checks,targets=targets,formula='E||CVmean - true_batch_mean||^2=(1/k-1/N)*S_residual^2; S uses N-1 denominator. Same fixed-batch token normalization multiplies both sides equally.',interpretation='Targets concern conditional raw-gradient sampling variance, not downstream accuracy or Adam/clipping. Small held-out diagnostic samples are estimates, not population guarantees. Oracle32 has zero within-batch sampling error, so nonzero residuals cannot match that variance exactly.'))
    log(dict(passed=True,targets=targets))

if __name__=='__main__':main()
