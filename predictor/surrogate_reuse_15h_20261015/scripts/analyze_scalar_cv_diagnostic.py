"""Derive an empirical scalar-rescaling bound from existing development metrics."""
from common import *
import csv


def main():
    source = EXP/'results/predictor_rollout_shift.json'
    diagnostic = json.loads(source.read_text())
    assert diagnostic['complete']
    rows = []
    for state in diagnostic['rows']:
        for path, metrics in state['predictors'].items():
            rho = metrics['centered_correlation']
            actual = metrics['centered_cv_ratio']
            assert np.isfinite(rho) and abs(rho) <= 1+1e-6
            best = max(0., 1-rho*rho)
            assert actual+1e-5 >= best
            rows.append(dict(state=state['state']['name'],
                             warmup_seed=state['state']['warmup_seed'],
                             steps=state['state']['step'],
                             predictor=Path(path).parent.name,
                             centered_correlation=rho,
                             actual_residual_variance_ratio=actual,
                             best_residual_ratio_with_one_scalar=best,
                             maximum_variance_reduction_percent=100*(1-best)))
    result = dict(complete=True, source_sha256=sha(source),
                  diagnostic_examples=diagnostic['n'], rows=rows,
                  identity='For centered true/predicted gradient matrices Gc/Pc with nonzero norms: min_a ||Gc-a Pc||_F^2 / ||Gc||_F^2 = 1-rho^2. The single scalar is unrestricted and fitted with all true gradients from this diagnostic population.',
                  scope='Posthoc CPU calculation from existing development-only gradient diagnostics. An empirical best-scalar bound, not a fitted/deployable method, new GPU experiment, batch-by-batch guarantee, heldout confirmation or downstream-accuracy bound. More expressive or example-dependent corrections are not bounded by this calculation. Pure-update mean bias is not removed.',
                  code_sha256=source_signature(__file__))
    write(EXP/'results/scalar_cv_diagnostic.json', result)
    with (EXP/'results/scalar_cv_diagnostic.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    log(dict(rows=len(rows), diagnostic_examples=diagnostic['n']))


if __name__ == '__main__':
    main()
