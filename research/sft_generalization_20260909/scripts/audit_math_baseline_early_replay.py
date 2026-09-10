"""Verify the full-test baseline against the previously stored1500 before later models finish."""
from common import *
root = S/'results/evaluation/math_reused5000/base'
end = min(DEADLINE, time.time()+1800)
while not (root/'math_summary.json').exists():
    if time.time() >= end: raise RuntimeError('FullMATH baseline readiness cutoff')
    time.sleep(5)
full = read(root/'math_predictions.jsonl'); assert len(full) == 5000
oldpath = S/'results/evaluation/math_reused1500/base/math_predictions.jsonl'
old = read(oldpath); validation = read(S/'data/math_reused1500.jsonl')
data = read(S/'data/math_reused5000.jsonl'); assert len(old) == len(validation) == 1500
fields = ['sample_hash', 'prediction', 'correct', 'finish_reason', 'generated_tokens']
different = []
for i,(reference,row) in enumerate(zip(old,validation)):
    j = row['source_test_index']
    assert row['problem'] == data[j]['problem'] and row['solution'] == data[j]['solution']
    if any(reference[k] != full[j][k] for k in fields):
        different.append(dict(validation_index=i, full_index=j,
                              differing_fields=[k for k in fields if reference[k] != full[j][k]]))
write(S/'audits/final_math_baseline_early_replay.json',dict(time=time.time(),
    plan_sha256=sha(S/'results/confirmation_plan.json'), script_sha256=sha(__file__),
    full_predictions_sha256=sha(root/'math_predictions.jsonl'), old_predictions_sha256=sha(oldpath),
    compared_rows=1500, compared_fields=fields, differences=different, passed=not different,
    scope='Early baseline-only check. Final all-model overlap replay remains required; historically reused data.'))
assert not different, different
print('FullMATH baseline matches all previously stored1500 outputs exactly.')
