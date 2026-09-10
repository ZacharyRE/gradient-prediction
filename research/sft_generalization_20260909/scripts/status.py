"""Small progress snapshot without printing token-by-token progress bars."""
from common import *
from datetime import datetime,timezone
print('UTC',datetime.now(timezone.utc).isoformat(),'hours remaining',round((DEADLINE-time.time())/3600,2))
jobs=json.loads((S/'results/train_queue.json').read_text());completed=[];active=[];pending=[]
for j in jobs:
 p=S/'results/training'/j['name']
 if (p/'complete.json').exists():completed.append(j['name'])
 elif p.exists():
  epochs=[json.loads(f.read_text())['epoch'] for f in p.glob('epoch*/ready.json')];active.append(dict(name=j['name'],epochs=sorted(epochs)))
 else:pending.append(j['name'])
print(json.dumps(dict(training_complete=len(completed),training_active=active,training_pending=len(pending),latest_completed=completed[-3:],failed=[f.name for f in (S/'logs').glob('failed_*.json')]),indent=2))
files=sorted((S/'results/evaluation/dev').glob('*/math_summary.json'),key=lambda p:p.stat().st_mtime)
print('Completed development evaluations',len(files))
for f in files[-6:]:
 d=json.loads(f.read_text());print(f.parent.name,d['correct'],d['samples'])
print('OOD evaluations',len(list((S/'results/evaluation').glob('ood_*/*/math_summary.json'))))
