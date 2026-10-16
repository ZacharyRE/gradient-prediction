"""Summarize registered graph/template and early-cutoff confirmations separately."""
from common import *
from collections import defaultdict
import csv
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--population',choices=['fullpool','candidate'],default='fullpool');a=ap.parse_args()
    from summarize_reuse import rows
    scopes={}
    configs=['position_candidate_confirmation_jobs'] if a.population=='candidate' else ['graph_confirmation_jobs','graph_subset8_control_jobs','position_confirmation_jobs','early_predictor_cutoff_jobs','early_template_jobs','early_original_jobs']
    for name in configs:
        path=EXP/'configs'/f'{name}.json';backend='native' if name.startswith('early_') else 'graph'
        for job in json.loads(path.read_text()):scopes[(backend,job['name'])]=job
    groups=defaultdict(dict)
    for r in rows:
        if r['subset']!=('candidate_0' if a.population=='candidate' else 'update') or r['warmup_seed'] not in [301,311,321]:continue
        root=Path(r['adapter']).parent if 'adapter' in r else None
        if root is None:
            # summarize_reuse exposes tags, while the manifest retains paths.
            root=EXP/'models'/r['tag'].rsplit('__step',1)[0]
        cfg_path=root/'config.json'
        if not cfg_path.exists():continue
        cfg=json.loads(cfg_path.read_text());backend=cfg.get('teacher_backend','native')
        if (backend,r['method']) not in scopes:continue
        key=(backend,r['method'],r['steps'],r['lr']);warm=r['warmup_seed']
        old=groups[key].get(warm)
        if old is not None:
            assert old['tag']==r['tag'] and old['accuracy']==r['accuracy'],('Ambiguous repeated trajectory',key,warm)
            continue
        groups[key][warm]=r
    result=[]
    for (backend,method,steps,lr),cohort in sorted(groups.items()):
        starts=sorted(cohort);rr=[cohort[w] for w in starts]
        result.append(dict(backend=backend,method=method,steps=steps,lr=lr,complete=starts==[301,311,321],warmup_seeds=starts,mean_accuracy_pct=float(np.mean([r['accuracy'] for r in rr])),accuracy_per_start=[r['accuracy'] for r in rr],mean_preparation_plus_online_seconds=float(np.mean([r['adaptation_seconds'] for r in rr])),seconds_per_start=[r['adaptation_seconds'] for r in rr],tags=[r['tag'] for r in rr]))
    stem='position_candidate_confirmation_summary' if a.population=='candidate' else 'backend_confirmation_summary'
    write(EXP/'results'/f'{stem}.json',dict(rows=result,population=a.population,scope='Only registered graph/template and early-cutoff methods. Backend is part of method identity; do not merge padded graph and native trajectories. Three starts required for complete cells. Repeatedly inspected accuracy_dev, not final outcomes.'))
    if result:
        with (EXP/'results'/f'{stem}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');writer.writeheader();writer.writerows(result)
    log(dict(cells=len(result),complete=sum(r['complete'] for r in result)))

if __name__=='__main__':main()
