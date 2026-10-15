"""Aggregate development curves by matched recipe; never inspect final tests."""
from common import *
import csv,collections

def main():
    rows=list(csv.DictReader((EXP/'results/development_live.csv').open()));groups=collections.defaultdict(list)
    for r in rows:
        if not r['dev_accuracy_pct']:continue
        groups[(r['task'],r['method'],float(r['lr']),int(r['steps']))].append(r)
    result=[]
    for (task,method,lr,steps),rr in groups.items():
        assert len({r['seed'] for r in rr})==len(rr)
        accuracy=[float(r['dev_accuracy_pct']) for r in rr]
        result.append(dict(task=task,method=method,lr=lr,steps=steps,n_seeds=len(rr),seeds=','.join(sorted(r['seed'] for r in rr)),dev_accuracy_mean=float(np.mean(accuracy)),dev_accuracy_seed_sd=float(np.std(accuracy,ddof=1)) if len(rr)>1 else None,online_seconds_mean=float(np.mean([float(r['online_seconds']) for r in rr])),prefix_seconds_mean=float(np.mean([float(r['prefix_preparation_seconds']) for r in rr])),tags=[r['tag'] for r in rr]))
    result.sort(key=lambda r:(r['task'],-r['n_seeds'],-r['dev_accuracy_mean'],r['online_seconds_mean']))
    write(EXP/'results/development_cohorts.json',dict(selection_only=True,scope='Online and prefix timings; inherited offline predictor preparation is not included in this working table.',rows=result))
    print(json.dumps(dict(checkpoint_cohorts=len(result),three_seed_cohorts=sum(r['n_seeds']>=3 for r in result))))

if __name__=='__main__':main()
