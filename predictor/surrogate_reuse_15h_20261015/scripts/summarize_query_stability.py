"""Describe query gradients and clipping; these are not causal attribution."""
from common import *
import csv


def main():
    development={r['tag']:r for r in csv.DictReader((EXP/'results/development_live.csv').open())}
    rows=[]
    for root in sorted((EXP/'models').glob('*__tquery_*')):
        if not (root/'summary.json').exists():continue
        cfg=json.loads((root/'config.json').read_text());job=cfg['job']
        history=json.loads((root/'history.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        for e in summary['entries']:
            hh=[h for h in history if h['step']<=e['steps']]
            norms=[h['gradient_norm'] for h in hh]
            rows.append(dict(task=e['task'],method=e['method'],seed=e['seed'],predictor_seed=e['predictor_seed'],lr=e['lr'],steps=e['steps'],query_probability=job['query_probability'],residual_ema=job['residual_ema'],clip_fraction=float(np.mean(np.array(norms)>1)),max_preclip_norm=max(norms),queries=sum(h['queried'] for h in hh),online_seconds=e['online_seconds'],dev_accuracy_pct=development.get(e['tag'],{}).get('dev_accuracy_pct',''),adapter_sha256=e['adapter_sha256']))
    write(EXP/'results/time_query_stability.json',dict(rows=rows,scope='Observed clipping and query counts through each saved checkpoint. Conditional raw-gradient unbiasedness does not imply unbiased clipping/Adam updates. Correlations with downstream collapse do not establish its cause. Predictor-only residual diagnostics omit the delayed residual baseline and do not measure variance of the entire query estimator.'))
    log(dict(checkpoints=len(rows)))


if __name__=='__main__':main()
