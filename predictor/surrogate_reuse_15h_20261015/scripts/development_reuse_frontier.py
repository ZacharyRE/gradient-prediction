"""Accuracy/time frontier on complete three-start development cells only."""
from common import *

def frontier(rows):
    def dominates(a,b):
        qa,qb=round(a['mean_accuracy_pct'],10),round(b['mean_accuracy_pct'],10)
        ta,tb=a['mean_preparation_plus_online_seconds'],b['mean_preparation_plus_online_seconds']
        return qa>=qb and ta<=tb and (qa>qb or ta<tb)
    return sorted([r for r in rows if not any(dominates(q,r) for q in rows)],key=lambda r:r['mean_preparation_plus_online_seconds'])

def main():
    development=json.loads((EXP/'results/development_summary.json').read_text())
    warm={r['warmup_seed']:r for r in development if r['method']=='warmup' and r['warmup_seed'] in [301,311,321]}
    assert set(warm)=={301,311,321}
    baseline=dict(method='warmup',steps=0,lr=None,complete=True,warmup_seeds=[301,311,321],mean_accuracy_pct=float(np.mean([warm[w]['accuracy'] for w in sorted(warm)])),mean_preparation_plus_online_seconds=0.,accuracy_per_start=[warm[w]['accuracy'] for w in sorted(warm)],seconds_per_start=[0.,0.,0.])
    result={}
    for population in ['candidate','fullpool']:
        path=EXP/'results'/f'{population}_confirmation_summary.json'
        if not path.exists():continue
        rows=[r for r in json.loads(path.read_text())['rows'] if r['complete']]+[baseline]
        result[population]=dict(cells=len(rows),frontier=frontier(rows),source_sha256=sha(path))
    write(EXP/'results/development_reuse_frontier.json',dict(populations=result,scope='Exploratory mean accuracy versus historical preparation+online time. Same3 development starts at every point. Warmup costs are common and excluded; model startup, offline predictor construction and evaluation are excluded. Frontier membership is descriptive, not a statistically confirmed advantage. Accuracy ties are rounded at1e-10 percentage points only to avoid floating-point sum-order artifacts.'))
    log({k:[(r['method'],r['steps'],r['mean_accuracy_pct'],r['mean_preparation_plus_online_seconds']) for r in v['frontier']] for k,v in result.items()})

if __name__=='__main__':main()
