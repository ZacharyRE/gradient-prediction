"""Check the unchanged native baseline prefix early while longer runs continue."""
from common import *
old = S/'results/native/dev/base/predictions.jsonl'
new = S/'results/native/final_dev500/base/predictions.jsonl'
reference = read(old); assert len(reference) == 128
end = min(DEADLINE, time.time()+900)
while time.time() < end:
    try:
        captured = new.read_bytes()
        rows = [json.loads(line) for line in captured.decode().splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(5); continue
    if len(rows) >= 128: break
    time.sleep(5)
else: raise RuntimeError('Native early-prefix readiness cutoff')
fields = ['index', 'sample_hash', 'prediction', 'correct', 'generated_tokens', 'finish_reason']
differences = [i for i,(a,b) in enumerate(zip(reference, rows[:128])) if any(a[k] != b[k] for k in fields)]
write(S/'audits/final_native_early128_replay.json', dict(time=time.time(),
    plan_sha256=sha(S/'results/confirmation_plan.json'), script_sha256=sha(__file__),
    old_predictions_sha256=sha(old), captured_new_file_sha256=hashlib.sha256(captured).hexdigest(),
    captured_new_rows=len(rows), prefix_rows=128, compared_fields=fields,
    differing_indices=differences, passed=not differences,
    scope='Early native baseline prefix check only; final full native analysis and protocol checks still required.'))
assert not differences, differences
print('Native baseline first128 exact replay passed.')
