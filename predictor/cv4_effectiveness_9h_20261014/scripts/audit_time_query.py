"""Exhaustively check the two query outcomes, independent of a trained model."""
from common import *

def main():
    rng=np.random.default_rng(42);worst=0
    for q in [.125,.25,.5,1.]:
        for _ in range(100):
            truth=rng.normal(size=100);base=rng.normal(size=100)*10
            expectation=(1-q)*base+q*(base+(truth-base)/q)
            worst=max(worst,float(np.max(np.abs(expectation-truth))))
    assert worst<1e-12
    write(EXP/'results/time_query_algebra_audit.json',dict(passed=True,max_absolute_expectation_error=worst,scope='Exhaustive Bernoulli outcomes conditional on fixed current batch/model and pre-query surrogate. First step separately uses exact truth. This says nothing about clipping/Adam bias or downstream quality.'))
    log(dict(passed=True,max_absolute_expectation_error=worst))

if __name__=='__main__':main()
