"""Descriptive standardization to shared case mix; no causal selection-bias claim."""
import argparse
import numpy as np
from common import *


def standardize(d, labels, selected, remaining, replicates=10000, seed=20261103):
    ns, n = d.shape
    labels = np.asarray(labels)
    selected = np.asarray(selected, dtype=int)
    remaining = np.asarray(remaining, dtype=int)
    assert len(set(selected) | set(remaining)) == n
    assert not set(selected) & set(remaining)
    categories = sorted(set(labels.tolist()))
    strata = []
    missing = []
    for label in categories:
        a = selected[labels[selected] == label]
        b = remaining[labels[remaining] == label]
        weight = (len(a)+len(b))/n
        row = dict(label=label, selected_n=len(a), remaining_n=len(b), full_weight=weight)
        if not len(a) or not len(b):
            missing.append(row)
        else:
            row.update(selected_per_seed_delta_pp=(d[:, a].mean(1)*100).tolist(),
                       remaining_per_seed_delta_pp=(d[:, b].mean(1)*100).tolist())
        strata.append((row, a, b))
    unadjusted = (d[:, selected].mean(1)-d[:, remaining].mean(1))*100
    result = dict(identifiable=not missing, missing_strata=missing,
                  unadjusted_per_seed_gap_pp=unadjusted.tolist(),
                  unadjusted_gap_pp=float(unadjusted.mean()), strata=[x[0] for x in strata])
    if missing:
        return result
    assert abs(sum(x[0]['full_weight'] for x in strata)-1) < 1e-12
    a_point = sum(row['full_weight']*d[:, a].mean(1) for row, a, b in strata)
    b_point = sum(row['full_weight']*d[:, b].mean(1) for row, a, b in strata)
    point = (a_point-b_point)*100
    rng = np.random.default_rng(seed)
    questions = []
    crossed = []
    for offset in range(0, replicates, 200):
        batch = min(200, replicates-offset)
        differences = np.zeros((batch, ns), dtype=float)
        for row, a, b in strata:
            ai = a[rng.integers(len(a), size=(batch, len(a)))]
            bi = b[rng.integers(len(b), size=(batch, len(b)))]
            differences += row['full_weight']*(d[:, ai].mean(2).T-d[:, bi].mean(2).T)
        questions.extend((differences.mean(1)*100).tolist())
        si = rng.integers(ns, size=(batch, ns))
        crossed.extend((np.take_along_axis(differences, si, axis=1).mean(1)*100).tolist())
    result.update(selected_standardized_per_seed_delta_pp=(a_point*100).tolist(),
                  remaining_standardized_per_seed_delta_pp=(b_point*100).tolist(),
                  standardized_per_seed_gap_pp=point.tolist(),
                  standardized_gap_pp=float(point.mean()),
                  unadjusted_minus_standardized_gap_pp=float(unadjusted.mean()-point.mean()),
                  question_ci95_pp=np.quantile(questions, [.025, .975]).tolist(),
                  seed_question_ci95_pp=np.quantile(crossed, [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoring', choices=['raw', 'strict'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    pp = S/'audits/validation_case_mix_plan.json'
    plan = json.loads(pp.read_text())
    primary_path = S/'results/confirmation_plan.json'
    primary = json.loads(primary_path.read_text())
    assert sha(primary_path) == plan['primary_plan_sha256']
    assert sha(S/'data/math_reused5000.jsonl') == plan['data_sha256']
    data = read(S/'data/math_reused5000.jsonl')
    remaining = np.array(primary['validation_excluded_sensitivity']['keep_indices'])
    selected = np.array(sorted(set(range(5000))-set(remaining)))
    assert len(remaining) == 3500 and len(selected) == 1500
    names = ['base']+[primary['families'][primary['primary_family']][str(k)] for k in plan['seeds']]
    strict_path = S/'audits/scoring_confirmation_math_reused5000.json'
    strict = json.loads(strict_path.read_text()) if args.scoring == 'strict' else None
    values = []
    identities = {}
    expected_hashes = None
    for name in names:
        root = S/'results/evaluation/math_reused5000'/name
        assert (root/'math_summary.json').exists()
        rows = read(root/'math_predictions.jsonl')
        assert len(rows) == 5000
        hashes = [r['sample_hash'] for r in rows]
        if expected_hashes is None:
            expected_hashes = hashes
        assert hashes == expected_hashes
        identities[name] = sha(root/'math_predictions.jsonl')
        if strict:
            assert strict['script_sha256'] == primary['strict_scoring_script_sha256']
            assert strict['models'][name]['predictions_sha256'] == identities[name]
            vector = strict['models'][name]['strict_vector']
        else:
            vector = [r['correct'] for r in rows]
        values.append(np.asarray(vector, dtype=int))
    d = np.stack(values[1:])-values[0]
    results = {}
    for fields in plan['stratifications']:
        labels = [' | '.join(row[field] for field in fields) for row in data]
        results['+'.join(fields)] = standardize(d, labels, selected, remaining,
            replicates=plan['bootstrap']['replicates'], seed=plan['bootstrap']['seed'])
    output = S/f'results/validation_case_mix_{args.scoring}.json'
    assert not output.exists()
    write(output, dict(time=time.time(), plan_sha256=sha(pp), script_sha256=sha(__file__),
        primary_plan_sha256=sha(primary_path), scoring=args.scoring, identities=identities,
        strict_audit_sha256=sha(strict_path) if strict else None,
        full5000_delta_pp=float(d.mean()*100), results=results, scope=plan['scope'],
        bootstrap_scope=plan['bootstrap']['conditional_scope']))
    print(json.dumps({k:{x:y for x,y in v.items() if x in ['identifiable','standardized_gap_pp',
        'unadjusted_gap_pp','question_ci95_pp','seed_question_ci95_pp']} for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
