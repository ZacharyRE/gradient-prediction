"""Descriptive paired comparisons with frozen cheaper controls; no selection."""
from common import *
from analyze_frozen_results import interval


def main():
    freeze=json.loads((EXP/'configs/final_freeze.json').read_text())
    comparisons={
        'paws':[('cv16','oracle16'),('anchor2','hold2'),('cv16','oracle_early')],
        'mnli':[('rls8','oracle8'),('cv8','oracle8'),('rls8','oracle_early'),('oracle_cached','oracle_quality'),('cv16','oracle_early')],
    }
    results=[]
    for split in ['oracle_confirmation','test']:
        for task,pairs in comparisons.items():
            source=read(data(task,split));ids=[r['sample_id'] for r in source]
            lookup={g:i for i,g in enumerate(sorted({r['group_id'] for r in source}))}
            inverse=np.array([lookup[r['group_id']] for r in source])
            folder=EXP/'results/eval'/freeze['evaluation_names'][task][split]
            values={}
            for entry in freeze['entries']:
                if entry['task']!=task or entry['cohort']=='warmup':continue
                pred=read(folder/(entry['tag']+'__'+task+'.jsonl'))
                assert [r['sample_id'] for r in pred]==ids
                values[entry['cohort'],entry['seed']]=np.array([r['correct'] for r in pred],dtype=float)
            for candidate,control in pairs:
                delta=np.stack([values[candidate,s]-values[control,s] for s in [101,102,103]])
                result=interval(delta,inverse)
                results.append(dict(task=task,split=split,candidate=candidate,control=control,
                    mean_gap_pp=result['mean_pct'],ci95_gap_pp=result['ci95_pct'],
                    per_seed_gap_pp={str(s):float(delta[i].mean()*100) for i,s in enumerate([101,102,103])},
                    n_examples=len(source),n_source_groups=len(lookup),bootstrap_draws=5000,
                    analysis_registration_phase='post_readout_secondary' if (task,candidate,control)==('mnli','cv16','oracle_early') else 'registered_during_frozen_readout'))
    write(EXP/'results/frozen_control_comparisons.json',dict(freeze_sha256=sha(EXP/'configs/final_freeze.json'),
        interpretation='Supplementary descriptive paired comparisons among already-frozen cohorts. The MNLI CV16-versus-early-Oracle comparison was added after readouts, and is explicitly post-readout exploratory analysis. No multiplicity correction or post-readout recipe selection. Early-Oracle comparisons change stopping point and learning rate; see the frozen plan. One shared warmup and historically inspected evaluation populations limit inference.',rows=results))
    log(dict(paired_control_comparisons=len(results)))


if __name__=='__main__':main()
