"""Exact discrete percentile-bootstrap distribution for four borderline paired contrasts."""
import numpy as np
from scipy.stats import binomtest
from common import *
gridpath=S/'results/objective_coefficient_grid.json';grid=json.loads(gridpath.read_text())
auditpath=S/'audits/scoring_objective_coefficient_grid.json';audit=json.loads(auditpath.read_text())
assert grid['scoring_audit_sha256']==sha(auditpath)
root=S/'results/evaluation/dev';control='teacher_all_dftkl0_lr5e5_s43_epoch4'
records=[]
for label in ['raw','strict']:
    vectors={}
    for name in [control,'teacher_all_dftkl01_lr5e5_s43_epoch4','teacher_all_dftkl1_lr5e5_s43_epoch4']:
        path=root/name/'math_predictions.jsonl';assert sha(path)==audit['models'][name]['predictions_sha256']
        rows=read(path);assert len(rows)==500
        vectors[name]=np.array([r['correct'] for r in rows] if label=='raw' else audit['models'][name]['strict_vector'],int)
    for coefficient in ['01','1']:
        name=f'teacher_all_dftkl{coefficient}_lr5e5_s43_epoch4';d=vectors[name]-vectors[control]
        counts=[int((d==value).sum()) for value in [-1,0,1]];n=len(d)
        mass=np.array([1.]);prob=np.array(counts,dtype=float)/n
        for _ in range(n):mass=np.convolve(mass,prob)
        assert abs(mass.sum()-1)<1e-11
        mass/=mass.sum();cdf=np.cumsum(mass)
        quantiles=(np.searchsorted(cdf,[.025,.975])-n)*100/n
        mc=next(r for r in grid['results'][label]['contrasts'] if r['kind']=='KL_minus_noKL' and r['objective']=='dft' and r['coefficient']==coefficient)
        wins,losses=counts[2],counts[0]
        records.append(dict(scoring=label,coefficient=coefficient,treatment=name,control=control,n=n,
            wins=wins,losses=losses,delta_pp=float(d.mean()*100),
            original_10000_draw_ci95_pp=mc['ci95_pp'],
            exact_discrete_percentile_bootstrap_ci95_pp=quantiles.tolist(),
            bootstrap_probability_delta_nonpositive=float(cdf[n]),
            exact_mcnemar_two_sided_p=binomtest(wins,wins+losses).pvalue))
write(S/'audits/objective_bootstrap_boundary.json',dict(time=time.time(),script_sha256=sha(__file__),
    objective_grid_sha256=sha(gridpath),scoring_audit_sha256=sha(auditpath),comparisons=records,
    method='For a single paired binary contrast, resampling500 question rows gives500 iid draws '
           'from the empirical ternary difference distribution. Convolve its three probabilities500 '
           'times, then invert the exact discrete CDF at.025/.975; no Monte Carlo sampling.',
    scope='Numerical boundary diagnosis, not a new efficacy test. Exact percentile bootstrap and exact '
          'conditional McNemar are different procedures and need not agree at discrete boundaries. '
          'All results single-seed/unadjusted; original analyses unchanged. No selection of the favorable interval.'))
print(json.dumps(records,indent=2))
