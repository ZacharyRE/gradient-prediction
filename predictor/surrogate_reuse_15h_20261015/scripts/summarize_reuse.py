"""Join completed development evaluations with exact rollout receipts."""
from common import *
import csv

rows=[]
for path in sorted((EXP/'results/eval').glob('*/summary.json')):
    protocol=json.loads((path.parent/'protocol.json').read_text());args=protocol['args']
    if args['split'] not in ['accuracy_dev']:continue
    entries=json.loads(Path(args['entries']).read_text());summary=json.loads(path.read_text())
    for entry in entries:
        key=entry['tag']+'__'+entry['task']
        if key not in summary:continue
        item=summary[key]
        rows.append(dict(evaluation=path.parent.name,task=entry['task'],warmup_seed=entry.get('warmup_seed'),subset=entry.get('subset'),method=entry['method'],seed=entry.get('seed'),steps=entry['steps'],lr=entry.get('lr'),accuracy=100*item['accuracy'],online_seconds=entry.get('online_seconds'),preparation_seconds=entry.get('cache_seconds'),adaptation_seconds=entry.get('online_seconds',0)+entry.get('cache_seconds',0),output_cap_hits=item['limit_hits'],tag=entry['tag']))
write(EXP/'results/development_summary.json',rows)
if rows:
    with (EXP/'results/development_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
log(dict(readouts=len(rows),max_accuracy=max((r['accuracy'] for r in rows),default=None)))
