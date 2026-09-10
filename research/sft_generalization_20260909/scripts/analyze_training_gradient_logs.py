"""Describe already-recorded preclip norms; no additional training or scoring."""
import csv
import math
import numpy as np
from common import *

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
pp = S/'audits/training_gradient_log_plan.json'
plan = json.loads(pp.read_text())
for path, digest in plan['source_sha256'].items():
    assert sha(S/path) == digest, path
torch_source = ROOT/'.venv/lib/python3.12/site-packages/torch/nn/utils/clip_grad.py'
assert 'clip_coef = max_norm / (total_norm + 1e-6)' in torch_source.read_text()
assert 'clip_coef_clamped = torch.clamp(clip_coef, max=1.0)' in torch_source.read_text()

def describe(rows):
    norms = np.asarray([x['gradient_norm'] for x in rows], dtype=float)
    finite = norms[np.isfinite(norms)]
    return dict(steps=len(rows), nonfinite_count=int((~np.isfinite(norms)).sum()),
        zero_count=int((norms == 0).sum()), min=float(finite.min()),
        median=float(np.median(finite)), p95=float(np.quantile(finite, .95)),
        max=float(finite.max()), above_one_count=int((norms > 1).sum()),
        epsilon_clipping_proxy_count=int((norms+1e-6 > 1).sum()),
        min_lr=min(x['lr'] for x in rows), max_lr=max(x['lr'] for x in rows),
        zero_lr_steps=sum(x['lr'] == 0 for x in rows),
        supervised_tokens=sum(x['supervised_tokens'] for x in rows))

results = {}
csv_rows = []
for name, endpoint in plan['models_stop_epoch'].items():
    root = S/'results/training'/name
    manifest = json.loads((root/'manifest.json').read_text())
    source = (root/'train_source.py').read_text()
    assert 'norm=torch.nn.utils.clip_grad_norm_(params,1.)' in source
    assert 'gradient_norm=float(norm)' in source
    assert manifest['arguments']['objective'] == 'token_mean'
    history = json.loads((root/'history.json').read_text())
    rows = [x for x in history if x['step'] and x['epoch'] <= endpoint]
    assert [x['step'] for x in rows] == list(range(1, len(rows)+1))
    assert sorted({x['epoch'] for x in rows}) == list(range(1, endpoint+1))
    exposure = json.loads((root/'exposure.json').read_text())[:len(rows)]
    assert [x['step'] for x in exposure] == [x['step'] for x in rows]
    assert sum(len(x['indices']) for x in exposure) == rows[-1]['seen']
    assert sum(x['supervised_tokens'] for x in rows) == rows[-1]['tokens']
    assert rows[-1]['seen'] == manifest['examples']*endpoint
    all_steps = describe(rows)
    assert all_steps['nonfinite_count'] == 0
    epochs = {str(k):describe([r for r in rows if r['epoch'] == k])
              for k in range(1, endpoint+1)}
    results[name] = dict(endpoint_epoch=endpoint, examples_seen=rows[-1]['seen'],
        all_steps=all_steps, per_epoch=epochs,
        arguments=manifest['arguments'], training_source_sha256=sha(root/'train_source.py'))
    csv_rows.append(dict(model=name, endpoint_epoch=endpoint, **all_steps))
output = S/'results/training_gradient_log_analysis.json'
assert not output.exists()
write(output, dict(time=time.time(), plan_sha256=sha(pp), script_sha256=sha(__file__),
    source_sha256=plan['source_sha256'], torch_clipping_source_sha256=sha(torch_source),
    post_outcome=True, results=results, scope=plan['scope'],
    numerical_limit='Logged norms are the preclip return value. The epsilon proxy uses recorded scalar norms; no near-boundary floating-point equivalence claim. A maximum safely below 1 excludes activation for that recorded run.'))
with (S/'results/training_gradient_log_analysis.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
print(json.dumps({n:{k:v['all_steps'][k] for k in ['steps','median','p95','max','epsilon_clipping_proxy_count']}
                  for n,v in results.items()}, indent=2))
