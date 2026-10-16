"""Descriptive three-seed development frontier; never an independent test."""
from common import *


def main():
    cohorts=json.loads((EXP/'results/development_cohorts.json').read_text())['rows']
    output=[]
    for task in ['paws','mnli']:
        rows=[dict(r,online_plus_prefix_seconds=r['online_seconds_mean']+r['prefix_seconds_mean']) for r in cohorts if r['task']==task and r['n_seeds']==3]
        for cost in ['online_seconds_mean','online_plus_prefix_seconds']:
            frontier=[]
            for r in sorted(rows,key=lambda r:r[cost]):
                dominated=any(s[cost]<=r[cost] and s['dev_accuracy_mean']>=r['dev_accuracy_mean'] and (s[cost]<r[cost] or s['dev_accuracy_mean']>r['dev_accuracy_mean']) for s in rows)
                if not dominated:frontier.append(r)
            output.append(dict(task=task,cost=cost,frontier=frontier))
    write(EXP/'results/development_frontier.json',dict(rows=output,scope='Descriptive screening only. Three paired update seeds per recipe and stopping point. Search timings require clean standalone replay; offline predictor preparation is excluded. No claim of statistical dominance, independent validation, or an optimal continuous-time frontier. All attempted checkpoints remain in experiment_index.csv.'))
    log(dict(frontiers=len(output)))


if __name__=='__main__':main()
