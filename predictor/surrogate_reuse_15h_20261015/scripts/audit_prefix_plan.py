"""Check a proposed sparse prefix cache against actual completed sampling logs."""
from reuse_train import needed_prefix_indices
from common import *

class Rows:
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)

checks=[]
for path in sorted((EXP/'models').glob('*/config.json')):
    cfg=json.loads(path.read_text())
    if not all(k in cfg for k in ['job','subset','update_seed','task']):continue
    history_path=path.parent/'history.json';summary_path=path.parent/'summary.json'
    if not history_path.exists() or not summary_path.exists():continue
    if not json.loads(summary_path.read_text()).get('complete'):continue
    job=cfg['job'];history=json.loads(history_path.read_text());ds=Rows(read(data(cfg['task'],cfg['subset'])))
    if len(history)!=job['steps']:continue
    expected=set(range(len(ds))) if job.get('table_bootstrap') else set()
    for row in history:
        ids=row['indices']
        if job['kind'] in ['cv','prediction','neural_hybrid']:expected.update(ids)
        elif row['true_gradient_examples']==32:expected.update(ids)
        elif row['true_gradient_examples']:
            positions=row.get('stratified_positions',row['positions']);expected.update(ids[j] for j in positions)
    planned=set(needed_prefix_indices([job],ds,cfg['update_seed']))
    assert expected==planned,(str(path),len(expected-planned),len(planned-expected))
    checks.append(dict(run=path.parent.name,kind=job['kind'],steps=job['steps'],planned_examples=len(planned),dataset_examples=len(ds)))
assert checks
report=dict(passed=True,checks=checks,scope='Exact prefix index coverage against completed sampling histories; GPU numerical and timing validation separate')
write(EXP/'results/prefix_plan_audit.json',report);log(dict(passed=True,completed_histories=len(checks),kinds=sorted(set(c['kind'] for c in checks))))
