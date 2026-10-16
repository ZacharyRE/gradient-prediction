"""Development-only candidate-size replication and prespecified Oracle choices."""
from common import *
from collections import defaultdict
import csv
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--population',choices=['candidate','fullpool'],default='candidate');a=ap.parse_args()
    from summarize_reuse import rows
    cohorts=defaultdict(dict)
    for r in rows:
        if r['subset']!=('candidate_0' if a.population=='candidate' else 'update') or r['warmup_seed'] not in [301,311,321]:continue
        prefixes=('candidate_reference_','state_candidate_') if a.population=='candidate' else ('fullpool_confirmation_','reference_w311','reference_w321')
        if not r['evaluation'].startswith(prefixes):continue
        key=(r['method'],r['steps'],r['lr'])
        if r['warmup_seed'] in cohorts[key]:
            prior=cohorts[key][r['warmup_seed']]
            assert prior['tag']==r['tag'] and prior['accuracy']==r['accuracy'],('Conflicting duplicate primary candidate readout',key)
            continue  # A clean identical rerun is not another independent seed.
        cohorts[key][r['warmup_seed']]=r
    out=[]
    for (method,steps,lr),rr in sorted(cohorts.items()):
        seeds=sorted(rr);x=[rr[w] for w in seeds]
        out.append(dict(method=method,steps=steps,lr=lr,complete=seeds==[301,311,321],warmup_seeds=seeds,mean_accuracy_pct=float(np.mean([r['accuracy'] for r in x])),accuracy_per_start=[r['accuracy'] for r in x],mean_preparation_plus_online_seconds=float(np.mean([r['adaptation_seconds'] for r in x])),seconds_per_start=[r['adaptation_seconds'] for r in x],tags=[r['tag'] for r in x]))
    choices={}
    # Method names are read from the registered grid, never guessed from a
    # favorable observed result. Only complete three-start cells are eligible.
    jobs=json.loads((EXP/'configs/candidate_development_reference_jobs.json').read_text())
    oracle_names={k:{j['name'] for j in jobs if j['kind'] in ['oracle','subset'] and j['k']==k} for k in [16,32]}
    assert all(len(names)==3 for names in oracle_names.values())
    for name,k,max_steps in [('independent_oracle32',32,512),('short_oracle32',32,128),('independent_oracle16',16,512)]:
        eligible=[r for r in out if r['complete'] and r['method'] in oracle_names[k] and r['steps']<=max_steps]
        expected=len(oracle_names[k])*len([s for s in [32,64,128,256,512] if s<=max_steps])
        if len(eligible)!=expected:choices[name]=dict(ready=False,complete_cells=len(eligible),expected_cells=expected);continue
        chosen=sorted(eligible,key=lambda r:(-r['mean_accuracy_pct'],r['mean_preparation_plus_online_seconds'],r['steps'],r['lr']))[0]
        choices[name]=dict(ready=True,chosen=chosen,rule='Highest three-start mean accuracy; ties lower mean preparation+online time, then steps, then LR')
    stem=a.population+'_confirmation_summary'
    write(EXP/'results'/f'{stem}.json',dict(rows=out,prospective_oracle_choices=choices,selection_rule_sha256=sha(EXP/'configs/development_selection_rule.json'),scope=f'Repeatedly inspected development only, {a.population}. Means with fewer than3 starts are explicitly incomplete. No final selection outcomes are read.'))
    if out:
        with (EXP/'results'/f'{stem}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
    log(dict(cells=len(out),complete_cells=sum(r['complete'] for r in out),oracle_choices=choices))

if __name__=='__main__':main()
