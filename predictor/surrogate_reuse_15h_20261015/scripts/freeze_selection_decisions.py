"""Choose candidates from selection_dev only, before any holdout readout."""
from common import *
from surrogate_freeze import FREEZE,DECISIONS,entry_key,critical_code
from collections import defaultdict


def main():
    assert FREEZE.exists() and not DECISIONS.exists()
    frozen=json.loads(FREEZE.read_text());assert frozen['complete'] and frozen['code']==critical_code();freeze_sha=sha(FREEZE)
    assert frozen['secondary_analysis_plan_sha256']==sha(EXP/'configs/secondary_analysis_plan.json'),'Secondary endpoints changed after recipe freeze'
    for root in ['eval','behavior']:
        for p in (EXP/'results'/root).glob('*/protocol.json'):
            assert json.loads(p.read_text())['args']['split']!='selection_holdout','Holdout readout preceded candidate-choice freeze'
    entries=[r['entry'] for r in frozen['entries'].values()];rows=read(data('mnli','selection_dev'));n=len(rows)
    assert {e['task'] for e in entries}=={'mnli'}
    targets=['all',*json.loads((EXP/'configs/secondary_analysis_plan.json').read_text())['secondary_targets']]
    masks={t:np.array([t=='all' or r.get('genre')==t for r in rows]) for t in targets};assert all(v.any() for v in masks.values())
    sources={};generated={};behavior=defaultdict(dict);teacher={}
    expected={entry_key(e):e for e in entries}
    def predictions(path):
        rr=read(path);assert len(rr)==n and all(p['sample_id']==r['sample_id'] and p['gold']==r['gold'] for p,r in zip(rr,rows));sources[str(path)]=sha(path)
        return rr
    for protocol_path in (EXP/'results/eval').glob('*/protocol.json'):
        protocol=json.loads(protocol_path.read_text())
        if protocol['args']['split']!='selection_dev' or protocol.get('surrogate_final_freeze',{}).get('freeze_sha256')!=freeze_sha:continue
        summary_path=protocol_path.parent/'summary.json'
        if not summary_path.exists():continue
        sources[str(protocol_path)]=sha(protocol_path);sources[str(summary_path)]=sha(summary_path)
        for key,s in json.loads(summary_path.read_text()).items():
            if key not in expected:continue
            rr=predictions(protocol_path.parent/f'{key}.jsonl');values=np.array([r['correct'] for r in rr],dtype=float);assert int(values.sum())==s['correct'] and s['n']==n
            if key in generated:assert np.array_equal(generated[key],values),'Conflicting duplicate evaluation'
            generated[key]=values
    assert set(generated)==set(expected),f'Missing generated selection_dev entries: {set(expected)-set(generated)}'
    declared_models={Path(p).parent.name for p in frozen.get('behavior_predictors',{})}
    if declared_models:
        for protocol_path in (EXP/'results/behavior').glob('*/protocol.json'):
            protocol=json.loads(protocol_path.read_text())
            if protocol['args']['split']!='selection_dev' or (protocol.get('freeze') or {}).get('freeze_sha256')!=freeze_sha:continue
            if not (protocol_path.parent/'complete.json').exists():continue
            sources[str(protocol_path)]=sha(protocol_path)
            for key,e in expected.items():
                for model in declared_models|{'teacher'}:
                    path=protocol_path.parent/f'{e["tag"]}__{model}.jsonl'
                    if not path.exists():continue
                    rr=predictions(path);correct=np.array([r['correct'] for r in rr],dtype=float);logits=np.array([r['logits'] for r in rr]);prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);goldprob=prob[np.arange(n),['ABC'.index(r['gold']) for r in rows]]
                    target=teacher if model=='teacher' else behavior[model]
                    current=dict(correct=correct,gold_probability=goldprob)
                    if key in target:assert all(np.array_equal(target[key][field],current[field]) for field in current),'Conflicting duplicate behavior/teacher evaluation'
                    target[key]=current
        assert set(teacher)==set(expected),'Exact forced-format teacher control is incomplete'
        for model in declared_models:assert set(behavior[model])==set(expected),f'Incomplete behavior readout: {model}'
    groups=defaultdict(list)
    for e in entries:
        subset=e.get('subset','')
        family='random' if subset.startswith('candidate_') else 'genre' if subset.startswith('genre_') else None
        if family is None:continue
        group=dict(family=family,warmup_seed=e['warmup_seed'],update_seed=e['seed'],method=e['method'],kind=e['kind'],steps=e['steps'],lr=e['lr'],loss_objective=e.get('loss_objective','full_answer'),true_every=e.get('true_every'))
        groups[json.dumps(group,sort_keys=True)].append(e)
    decisions=[]
    for key,ee in sorted(groups.items()):
        group=json.loads(key);expected_subsets={f'candidate_{i}' for i in range(5)} if group['family']=='random' else {f'genre_{g}' for g in ['fiction','government','slate','telephone','travel']}
        assert {e['subset'] for e in ee}==expected_subsets and len(ee)==5,group
        for target,mask in masks.items():
            selectors={'generated_accuracy':{entry_key(e):generated[entry_key(e)] for e in ee}}
            if declared_models:
                selectors['exact_teacher_forced_accuracy']={entry_key(e):teacher[entry_key(e)]['correct'] for e in ee}
                selectors['exact_teacher_gold_probability']={entry_key(e):teacher[entry_key(e)]['gold_probability'] for e in ee}
                for model in sorted(declared_models):
                    selectors[f'{model}:forced_accuracy']={entry_key(e):behavior[model][entry_key(e)]['correct'] for e in ee}
                    selectors[f'{model}:gold_probability']={entry_key(e):behavior[model][entry_key(e)]['gold_probability'] for e in ee}
            for selector,values in selectors.items():
                scores=[dict(subset=e['subset'],tag=e['tag'],score=float(values[entry_key(e)][mask].mean())) for e in ee];scores.sort(key=lambda x:(-x['score'],x['subset']))
                decisions.append(dict(**group,target=target,selector=selector,selection_examples=int(mask.sum()),selected_subset=scores[0]['subset'],selected_tag=scores[0]['tag'],candidate_scores=scores))
    assert decisions
    record=dict(complete=True,frozen_at=time.time(),freeze_sha256=freeze_sha,source_sha256=source_signature(__file__),selection_data_sha256=sha(data('mnli','selection_dev')),secondary_analysis_plan_sha256=sha(EXP/'configs/secondary_analysis_plan.json'),readout_sources=sources,decisions=decisions,no_holdout_outcomes_used=True,tie_rule='Lexicographic subset id',meaning='These fixed candidate choices are evaluated using Oracle and surrogate models on selection_holdout; no model recipe or stopping point changes are permitted')
    with DECISIONS.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    log(dict(decisions=len(decisions),decision_sha256=sha(DECISIONS)))

if __name__=='__main__':main()
