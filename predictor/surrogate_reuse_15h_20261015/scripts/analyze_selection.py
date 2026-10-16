"""Evaluate frozen candidate choices; never select a recipe from holdout results."""
from common import *
from surrogate_freeze import FREEZE,DECISIONS,entry_key
from collections import defaultdict
import argparse,csv
from functools import lru_cache

@lru_cache(maxsize=8)
def bootstrap_weights(starts,counts_tuple,seed,nboot):
    counts=np.array(counts_tuple);G=len(counts);rng=np.random.default_rng(seed)
    sw=np.stack([np.bincount(rng.integers(starts,size=starts),minlength=starts) for _ in range(nboot)])
    gw=np.stack([np.bincount(rng.integers(G,size=G),minlength=G) for _ in range(nboot)])
    return sw,gw,starts*(gw@counts)

def rank_correlation(x,y):
    def ranks(z):
        z=np.asarray(z);return np.array([np.flatnonzero(np.sort(z)==v).mean() for v in z])
    x=ranks(x);y=ranks(y)
    if x.std()==0 or y.std()==0:return None
    return float(np.corrcoef(x,y)[0,1])

def paired_interval(values,groups,seed=887,nboot=4000):
    """Crossed start/group bootstrap, conditional on already frozen dev choices."""
    values=np.asarray(values,dtype=float);assert values.ndim==2 and values.shape[1]==len(groups)
    _,group_index=np.unique(groups,return_inverse=True);G=int(group_index.max()+1);S=values.shape[0];counts=np.bincount(group_index,minlength=G)
    sums=np.stack([np.bincount(group_index,weights=row,minlength=G) for row in values]);sw,gw,denominator=bootstrap_weights(S,tuple(map(int,counts)),seed,nboot)
    samples=np.einsum('bs,sg,bg->b',sw,sums,gw,optimize=True)/denominator
    return dict(mean=float(values.mean()),ci95=list(map(float,np.quantile(samples,[.025,.975]))),per_start=list(map(float,values.mean(1))),starts=S,source_groups=G,examples=values.shape[1],bootstrap_repetitions=nboot,scope='Paired crossed start/source-group percentile bootstrap, conditional on frozen dev choices; three starts provide limited evidence and no automatic noninferiority claim')

def read_frozen_generated(split,frozen,bundle=None):
    expected={k:v['entry'] for k,v in frozen['entries'].items()};rows=read(data('mnli',split));values={};sources={}
    if bundle is not None:
        with np.load(bundle,allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata']));assert metadata['freeze_sha256']==sha(FREEZE) and metadata['decision_sha256']==sha(DECISIONS)
            assert metadata['data_sha256'][split]==sha(data('mnli',split))
            tags=list(map(str,z['tags']));matrix=z[split];assert set(tags)==set(expected) and matrix.shape==(len(tags),len(rows)) and matrix.dtype==np.bool_
            assert metadata['sample_ids'][split]==[r['sample_id'] for r in rows]
            values={tag:matrix[i].astype(float) for i,tag in enumerate(tags)}
            assert [int(matrix[i].sum()) for i in range(len(tags))]==metadata['correct_counts'][split]
        return rows,values,{split+'_prediction_bundle_sha256':sha(bundle),split+'_original_sources':metadata['sources'][split]}
    for protocol_path in (EXP/'results/eval').glob('*/protocol.json'):
        protocol=json.loads(protocol_path.read_text())
        if protocol['args']['split']!=split or (protocol.get('surrogate_final_freeze') or {}).get('freeze_sha256')!=sha(FREEZE):continue
        summary_path=protocol_path.parent/'summary.json'
        if not summary_path.exists():continue
        summary=json.loads(summary_path.read_text());sources[str(protocol_path)]=sha(protocol_path);sources[str(summary_path)]=sha(summary_path)
        for key in set(summary)&set(expected):
            path=protocol_path.parent/f'{key}.jsonl';rr=read(path)
            assert len(rr)==len(rows) and all(a['sample_id']==b['sample_id'] and a['gold']==b['gold'] for a,b in zip(rr,rows))
            v=np.array([r['correct'] for r in rr],dtype=float);assert int(v.sum())==summary[key]['correct']
            if key in values:assert np.array_equal(values[key],v)
            values[key]=v;sources[str(path)]=sha(path)
    assert set(values)==set(expected),f'Missing {split} evaluations: {set(expected)-set(values)}'
    return rows,values,sources

def direct_quality(entries,values,rows,oracle,starts):
    """Paired quality over five fixed candidates or the single full update pool."""
    groups=defaultdict(list);warmups={}
    for e in entries.values():
        if e.get('steps')==0 and e.get('warmup_seed') in starts:
            warmups[e['warmup_seed']]=values[entry_key(e)]
        subset=e.get('subset','');family='random' if subset.startswith('candidate_') else 'genre' if subset.startswith('genre_') else 'fullpool' if subset=='update' else None
        if family is None:continue
        key=dict(family=family,method=e['method'],kind=e['kind'],steps=e['steps'],lr=e['lr'],loss_objective=e.get('loss_objective','full_answer'),true_every=e.get('true_every'))
        groups[json.dumps(key,sort_keys=True)].append(e)
    result=[];source_groups=[r['group_id'] for r in rows]
    for key,ee in sorted(groups.items()):
        means=[];reference_means=[];per_candidate=[]
        for warm in starts:
            candidates=sorted((e for e in ee if e['warmup_seed']==warm),key=lambda e:e['subset']);assert len(candidates)==(1 if json.loads(key)['family']=='fullpool' else 5)
            actual=np.stack([values[entry_key(e)] for e in candidates]);reference=np.stack([values[entry_key(oracle[(warm,e['subset'])])] for e in candidates])
            means.append(actual.mean(0));reference_means.append(reference.mean(0))
            per_candidate.extend(dict(warmup_seed=warm,subset=e['subset'],accuracy=float(v.mean()),oracle_reference_accuracy=float(o.mean())) for e,v,o in zip(candidates,actual,reference))
        means=np.stack(means);reference_means=np.stack(reference_means)
        record=dict(**json.loads(key),accuracy=paired_interval(means,source_groups),delta_vs_oracle_same_candidate=paired_interval(means-reference_means,source_groups),per_candidate=per_candidate,scope='Five fixed candidates averaged within each start, or the single full update pool; paired start/source-group uncertainty. Candidate subsets are fixed, not treated as independent samples. Each population uses its own development-selected Oracle reference when specified.')
        if set(warmups)==set(starts):record['delta_vs_warmup']=paired_interval(means-np.stack([warmups[w] for w in starts]),source_groups)
        result.append(record)
    return result

def paired_recipe_comparisons(entries,values,rows,starts,specifications):
    source_groups=[r['group_id'] for r in rows];results=[]
    def family(e):
        subset=e.get('subset','');return 'fullpool' if subset=='update' else 'random' if subset.startswith('candidate_') else 'genre' if subset.startswith('genre_') else None
    for spec in specifications:
        actual=[];reference=[]
        for warm in starts:
            selected=[]
            for side in ['actual','reference']:
                ee=[e for e in entries.values() if e.get('warmup_seed')==warm and family(e)==spec['family'] and all(e.get(k)==v for k,v in spec[side].items())]
                assert len(ee)==(1 if spec['family']=='fullpool' else 5),(spec,warm,side)
                selected.append({e['subset']:e for e in ee});assert len(selected[-1])==len(ee)
            assert set(selected[0])==set(selected[1]);subsets=sorted(selected[0])
            actual.append(np.mean([values[entry_key(selected[0][subset])] for subset in subsets],axis=0));reference.append(np.mean([values[entry_key(selected[1][subset])] for subset in subsets],axis=0))
        actual=np.stack(actual);reference=np.stack(reference)
        results.append(dict(**spec,actual_accuracy=paired_interval(actual,source_groups),reference_accuracy=paired_interval(reference,source_groups),delta=paired_interval(actual-reference,source_groups)))
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis-spec',required=True);ap.add_argument('--prediction-bundle',type=Path);a=ap.parse_args();spec_path=Path(a.analysis_spec)
    frozen=json.loads(FREEZE.read_text());decisions=json.loads(DECISIONS.read_text());assert frozen['complete'] and decisions['complete'] and decisions['freeze_sha256']==sha(FREEZE)
    # The reference and stopping point must be part of the pre-readout freeze.
    if a.prediction_bundle:assert sha(spec_path) in {r['sha256'] for r in frozen['recipes'].values()},'Reproduction must use the exact frozen analysis spec'
    else:assert str(spec_path.resolve()) in frozen['recipes'] and frozen['recipes'][str(spec_path.resolve())]['sha256']==sha(spec_path)
    spec=json.loads(spec_path.read_text());reference=spec['oracle_reference'];entries={k:v['entry'] for k,v in frozen['entries'].items()};by_tag={e['tag']:e for e in entries.values()}
    rows,holdout,sources=read_frozen_generated('selection_holdout',frozen,a.prediction_bundle);devrows,dev,devsources=read_frozen_generated('selection_dev',frozen,a.prediction_bundle);sources.update(devsources)
    oracle={}
    def is_reference(e):
        population='fullpool' if e.get('subset')=='update' else 'candidate'
        selector=spec.get('oracle_reference_by_population',{}).get(population,reference)
        return all(e.get(k)==v for k,v in selector.items())
    for e in entries.values():
        if not is_reference(e):continue
        key=(e['warmup_seed'],e.get('subset'));assert key not in oracle;oracle[key]=e
    oracle_choices={(d['family'],d['warmup_seed'],d['target']):d for d in decisions['decisions'] if is_reference(d) and d['selector']=='generated_accuracy'}
    groups=defaultdict(list)
    for d in decisions['decisions']:
        signature={k:d[k] for k in ['family','method','kind','steps','lr','loss_objective','true_every','target','selector']}
        groups[json.dumps(signature,sort_keys=True)].append(d)
    results=[];detail=[]
    for signature,dd in sorted(groups.items()):
        dd.sort(key=lambda d:d['warmup_seed']);assert [d['warmup_seed'] for d in dd]==spec['heldout_warmups']
        group=json.loads(signature);target=group['target'];mask=np.array([target=='all' or r.get('genre')==target for r in rows]);devmask=np.array([target=='all' or r.get('genre')==target for r in devrows]);source_groups=np.array([r['group_id'] for r in rows])[mask]
        chosen=[];chosen_reference=[];random=[];ceiling=[];direct=[];ranks=[];fidelity_ranks=[];dev_ranks=[];agreement=[];metadata=[];initial_choice=[];train_all=[]
        for d in dd:
            warm=d['warmup_seed'];scores=sorted(d['candidate_scores'],key=lambda x:x['subset']);subsets=[x['subset'] for x in scores]
            oracle_vectors=np.stack([holdout[entry_key(oracle[(warm,s)])][mask] for s in subsets]);oracle_dev=np.array([dev[entry_key(oracle[(warm,s)])][devmask].mean() for s in subsets]);oracle_acc=oracle_vectors.mean(1)
            selected_index=subsets.index(d['selected_subset']);baseline=oracle_choices[(d['family'],warm,target)];baseline_index=subsets.index(baseline['selected_subset']);selected=oracle_vectors[selected_index];baseline_values=oracle_vectors[baseline_index]
            surrogate_vectors=np.stack([holdout[entry_key(by_tag[x['tag']])][mask] for x in scores]);surrogate_acc=surrogate_vectors.mean(1)
            chosen.append(selected);chosen_reference.append(baseline_values);random.append(oracle_vectors.mean(0));ceiling.append(oracle_vectors[int(oracle_acc.argmax())]);direct.append(surrogate_vectors[selected_index]);initial_choice.append(oracle_vectors[0])
            rank=rank_correlation([x['score'] for x in scores],oracle_acc);ranks.append(rank);fidelity_ranks.append(rank_correlation(surrogate_acc,oracle_acc));dev_ranks.append(rank_correlation([x['score'] for x in scores],oracle_dev));agreement.append(d['selected_subset']==baseline['selected_subset'])
            if d['family']=='genre' and target!='all':metadata.append(oracle_vectors[subsets.index('genre_'+target)])
            if d['family']=='random' and spec.get('random_family_train_all_control'):
                candidates=[e for e in entries.values() if e.get('warmup_seed')==warm and all(e.get(k)==v for k,v in spec['random_family_train_all_control'].items())];assert len(candidates)==1
                train_all.append(holdout[entry_key(candidates[0])][mask])
            detail.append(dict(**group,warmup_seed=warm,selected_subset=d['selected_subset'],oracle_dev_selected_subset=baseline['selected_subset'],selected_oracle_accuracy=float(selected.mean()),oracle_dev_selected_accuracy=float(baseline_values.mean()),random_oracle_accuracy=float(oracle_vectors.mean()),best_oracle_holdout_accuracy=float(oracle_acc.max()),direct_deploy_generated_accuracy=float(surrogate_vectors[selected_index].mean()),dev_score_vs_oracle_holdout_spearman=rank,oracle_candidate_spread=float(oracle_acc.max()-oracle_acc.min()),candidate_scores=[dict(**x,oracle_holdout_accuracy=float(oa),surrogate_generated_holdout_accuracy=float(sa)) for x,oa,sa in zip(scores,oracle_acc,surrogate_acc)]))
        chosen=np.stack(chosen);baseline=np.stack(chosen_reference);random=np.stack(random);ceiling=np.stack(ceiling);direct=np.stack(direct)
        r=dict(**group,selection_oracle_accuracy=paired_interval(chosen,source_groups),oracle_dev_selection_accuracy=paired_interval(baseline,source_groups),selection_delta_vs_oracle_dev_choice=paired_interval(chosen-baseline,source_groups),selection_delta_vs_random=paired_interval(chosen-random,source_groups),selection_delta_vs_constant_initial_choice=paired_interval(chosen-np.stack(initial_choice),source_groups),regret_to_best_oracle_holdout_ceiling=paired_interval(ceiling-chosen,source_groups),direct_deploy_accuracy=paired_interval(direct,source_groups),direct_deploy_delta_vs_oracle_dev_choice=paired_interval(direct-baseline,source_groups),top_choice_agreement=float(np.mean(agreement)),dev_score_vs_oracle_holdout_spearman=ranks,dev_score_vs_oracle_dev_spearman=dev_ranks,generated_model_fidelity_spearman=fidelity_ranks)
        if metadata:r['selection_delta_vs_matching_genre_metadata']=paired_interval(chosen-np.stack(metadata),source_groups)
        if train_all:
            r['train_all_original_pool_accuracy']=paired_interval(np.stack(train_all),source_groups)
            r['selection_delta_vs_train_all_original_pool']=paired_interval(chosen-np.stack(train_all),source_groups)
        results.append(r)
    quality=direct_quality(entries,holdout,rows,oracle,spec['heldout_warmups'])
    paired_controls=paired_recipe_comparisons(entries,holdout,rows,spec['heldout_warmups'],spec.get('direct_paired_controls',[]))
    result=dict(complete=True,freeze_sha256=sha(FREEZE),decisions_sha256=sha(DECISIONS),spec=spec,sources=sources,groups=results,per_start=detail,direct_quality=quality,direct_paired_comparisons=paired_controls,source_sha256=source_signature(__file__),units='Accuracies and gaps are fractions; multiply by100 for percentage points',important='Candidate recipes and choices were frozen before holdout readouts. Best-holdout-candidate ceiling is nondeployable. Secondary genre/readout analyses are descriptive. No best method is selected from these results.')
    write(EXP/'results/final_selection_analysis.json',result)
    flat=[]
    for r in results:
        row={k:v for k,v in r.items() if not isinstance(v,(list,dict))}
        for k,v in r.items():
            if isinstance(v,dict) and 'ci95' in v:row.update({k+'_pct':100*v['mean'],k+'_ci_low':100*v['ci95'][0],k+'_ci_high':100*v['ci95'][1]})
        flat.append(row)
    keys=list(dict.fromkeys(k for row in flat for k in row))
    with (EXP/'results/final_selection_analysis.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(flat)
    log(dict(groups=len(results),per_start_rows=len(detail),oracle_reference=reference))

if __name__=='__main__':main()
