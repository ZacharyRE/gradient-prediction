"""Paired seed/source-group analysis of preregistered final readouts.

Requires final_freeze.json with cohort annotations, comparison_reference per
task, and evaluation_names[task][split]. No recipe selection or mutation occurs.
"""
from common import *
import argparse,collections,csv

def interval(values,group_inverse,draws=5000,seed_id=1729):
    """Resample whole source groups and update seeds; preserve paired differences."""
    values=np.asarray(values,dtype=np.float64)
    nseeds,n=values.shape;ngroups=int(group_inverse.max())+1
    sums=np.stack([np.bincount(group_inverse,weights=row,minlength=ngroups) for row in values])
    sizes=np.bincount(group_inverse,minlength=ngroups);rng=np.random.default_rng(seed_id);samples=[]
    for start in range(0,draws,128):
        b=min(128,draws-start)
        sw=rng.multinomial(nseeds,np.full(nseeds,1/nseeds),size=b)/nseeds
        gw=rng.multinomial(ngroups,np.full(ngroups,1/ngroups),size=b)
        samples.extend(((sw@sums*gw).sum(1)/(gw@sizes)).tolist())
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(mean_pct=float(values.mean()*100),ci95_pct=[float(lo*100),float(hi*100)],n_seeds=nseeds,n_examples=n,n_source_groups=ngroups,bootstrap_draws=draws)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['oracle_confirmation','test'],required=True);ap.add_argument('--bootstrap',type=int,default=5000);a=ap.parse_args()
    path=EXP/'configs/final_freeze.json';freeze=json.loads(path.read_text());results=[];comparisons=[];receipts=[]
    for task in sorted({e['task'] for e in freeze['entries']}):
        source=read(data(task,a.split));ids=[r['sample_id'] for r in source]
        groups=[r['group_id'] for r in source];lookup={g:i for i,g in enumerate(sorted(set(groups)))};inverse=np.array([lookup[g] for g in groups])
        folder=EXP/'results/eval'/freeze['evaluation_names'][task][a.split]
        protocol=json.loads((folder/'protocol.json').read_text());assert protocol['args']['split']==a.split
        assert protocol['full_inputs'] and protocol['input_token_ids_verified_at_generation']
        assert protocol['input_sha256'][task]==sha(data(task,a.split))
        cohorts=collections.defaultdict(dict);cohort_entries=collections.defaultdict(list)
        for e in [e for e in freeze['entries'] if e['task']==task]:
            predfile=folder/(e['tag']+'__'+task+'.jsonl');pred=read(predfile)
            assert [r['sample_id'] for r in pred]==ids,(task,e['tag'],'Readout is incomplete or misaligned')
            assert protocol['adapter_sha256'][e['adapter']]==e['adapter_sha256']
            cohort=e['cohort'];seed_id=int(e['seed']);assert seed_id not in cohorts[cohort]
            cohorts[cohort][seed_id]=np.array([bool(r['correct']) for r in pred],dtype=np.float64)
            cohort_entries[cohort].append(e)
            receipts.append(dict(task=task,tag=e['tag'],split=a.split,n=len(pred),accuracy_pct=100*sum(r['correct'] for r in pred)/len(pred),strict_accuracy_pct=100*sum(r['strict_correct'] for r in pred)/len(pred),generation_limit_hits=sum(r['finish_reason']=='length' for r in pred),prediction_sha256=sha(predfile)))
        reference_name=freeze['comparison_reference'][task];reference=cohorts[reference_name]
        for cohort,items in cohorts.items():
            ordered=sorted(items);matrix=np.stack([items[s] for s in ordered]);record=interval(matrix,inverse,a.bootstrap)
            ee=cohort_entries[cohort]
            record.update(task=task,cohort=cohort,seeds=ordered,per_seed_accuracy_pct={str(s):float(items[s].mean()*100) for s in ordered},online_seconds_mean=float(np.mean([e.get('online_seconds',0) for e in ee])),prefix_seconds_mean=float(np.mean([e.get('cache_seconds',0) for e in ee])),primary_candidate=any(e.get('primary_candidate',False) for e in ee))
            results.append(record)
            if cohort==reference_name:continue
            if set(items)==set(reference):
                delta=np.stack([items[s]-reference[s] for s in sorted(reference)])
            elif cohort=='warmup' and len(items)==1:
                only=next(iter(items.values()));delta=np.stack([only-reference[s] for s in sorted(reference)])
            else:
                comparisons.append(dict(task=task,cohort=cohort,reference=reference_name,paired=False,reason='Seed sets differ; no paired claim'))
                continue
            ci=interval(delta,inverse,a.bootstrap)
            comparisons.append(dict(task=task,cohort=cohort,reference=reference_name,paired=True,mean_gap_pp=ci['mean_pct'],ci95_gap_pp=ci['ci95_pct'],per_seed_gap_pp={str(s):float(delta[i].mean()*100) for i,s in enumerate(sorted(reference))},n_seeds=len(reference),n_source_groups=len(lookup),point_within_1pp=ci['mean_pct']>=-1,ci_lower_above_minus1pp=ci['ci95_pct'][0]>-1,primary_candidate=record['primary_candidate']))
    output=dict(split=a.split,freeze_sha256=sha(path),results=results,comparisons=comparisons,receipts=receipts,interpretation='Descriptive paired source-group/update-seed bootstrap, conditional on one fixed warmup and the recorded finite seeds. Test populations have been inspected in prior studies. Point agreement within1pp alone is not evidence of noninferiority. Timings in this accuracy table are original training receipts; use separately audited standalone timing replays for final speed claims.')
    write(EXP/'results'/f'frozen_{a.split}_analysis.json',output)
    log(dict(split=a.split,cohorts=len(results),comparisons=len(comparisons)))

if __name__=='__main__':main()
