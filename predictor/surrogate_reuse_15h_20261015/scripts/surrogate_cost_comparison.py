"""Account for frozen deployment recipes without treating component sums as pipelines."""
from common import *
from surrogate_freeze import FREEZE,DECISIONS,entry_key
from collections import defaultdict
import csv,math

def main():
    frozen=json.loads(FREEZE.read_text());decisions=json.loads(DECISIONS.read_text());analysis=json.loads((EXP/'results/final_selection_analysis.json').read_text());policy=json.loads((EXP/'configs/surrogate_cost_policy.json').read_text());catalog=json.loads((EXP/'results/surrogate_preparation_components.json').read_text())
    assert analysis['freeze_sha256']==sha(FREEZE)==decisions['freeze_sha256']
    entries={r['entry']['tag']:r['entry'] for r in frozen['entries'].values()};models={r['predictor']:r for r in catalog['predictors']}
    timing_path=EXP/'results/standalone_reuse_timing_summary.json';timings={r['tag']:r for r in json.loads(timing_path.read_text())['rows']} if timing_path.exists() else {}
    genroot=EXP/'results/eval/surrogate_final_selection_dev';generated=json.loads((genroot/'summary.json').read_text());runtime=json.loads((genroot/'runtime_receipt.json').read_text())
    behaviorroot=EXP/'results/behavior/surrogate_final_behavior_selection_dev';behavior=json.loads((behaviorroot/'summary.json').read_text());bp=json.loads((behaviorroot/'protocol.json').read_text())
    def ready(e):return e.get('cache_seconds',0)+e.get('online_seconds',0)
    def model_names(e):return [Path(e['predictor_path']).parent.name] if e.get('predictor_path') else []
    def offline(names):
        names=sorted(set(names));components=set();warmups=set()
        for name in names:
            assert name in models,name
            components.update(models[name]['components']);warmups.update(models[name]['source_warmup_seeds'])
        cost=sum(catalog['components'][k]['seconds'] for k in components);source=sum(catalog['source_warmups'][str(w)]['seconds'] for w in warmups)
        return dict(predictors=names,component_ids=sorted(components),source_warmups=sorted(warmups),recorded_preparation_seconds_excluding_source_warmups=cost,recorded_source_warmup_seconds=source,recorded_component_sum_seconds=cost+source)
    def reference(e):
        population='fullpool' if e.get('subset')=='update' else 'candidate';selector=analysis['spec'].get('oracle_reference_by_population',{}).get(population,analysis['spec']['oracle_reference'])
        found=[x for x in entries.values() if x.get('warmup_seed')==e.get('warmup_seed') and x.get('subset')==e.get('subset') and all(x.get(k)==v for k,v in selector.items())];assert len(found)==1
        return found[0]
    def comparable_identity(a,b):
        return a['adapter_sha256']==b['adapter_sha256'] and all(a.get(k)==b.get(k) for k in ['warmup_seed','subset','steps','lr','kind','k'])
    def readout(ee,selector,warm):
        if selector=='generated_accuracy':
            times=[generated[entry_key(e)].get('original_generation_seconds',generated[entry_key(e)]['seconds']) for e in ee]
            return dict(candidate_seconds=sum(times),shared_seconds=0.,total_seconds=sum(times),kind='generated',additional_shared_engine_initialization_seconds=runtime.get('engine_initialization_seconds'))
        teacher=selector.startswith('exact_teacher_');model=next(iter(behavior[ee[0]['tag']])) if teacher else selector.split(':')[0]
        rr=[behavior[e['tag']][model] for e in ee]
        if teacher:
            candidate=sum(r['adapter_load_seconds']+r['current_hidden_seconds']+r['teacher_forced_seconds'] for r in rr);anchor=0.
        else:
            candidate=sum(r['adapter_load_seconds']+r['seconds'] for r in rr);anchor=bp['initial_teacher_anchors'][str(warm)]['seconds'] if 'one_time_initial_teacher_seconds' in rr[0] else 0.
        shared=bp['cache_seconds']+bp['metadata_seconds']+anchor
        return dict(candidate_seconds=candidate,shared_seconds=shared,total_seconds=candidate+shared,kind='exact_forced_teacher' if teacher else 'small_behavior_model',additional_shared_engine_initialization_seconds=None)
    selection=[]
    for d in decisions['decisions']:
        ee=[entries[x['tag']] for x in d['candidate_scores']];chosen=entries[d['selected_tag']];ref=reference(chosen);ro=readout(ee,d['selector'],d['warmup_seed']);already_trained=any(comparable_identity(e,ref) for e in ee)
        names=[n for e in ee for n in model_names(e)]
        if ':' in d['selector']:names.append(d['selector'].split(':')[0])
        training=sum(ready(e) for e in ee);final=0. if already_trained else ready(ref)
        selection.append(dict(**{k:d[k] for k in ['family','warmup_seed','method','kind','steps','lr','target','selector','selected_subset']},candidate_training_component_seconds=training,selection_readout=ro,final_reference_training_component_seconds=final,chosen_reference_already_trained=already_trained,ready_component_sum_seconds=training+ro['total_seconds']+final,offline=offline(names),scope='Five candidate adaptations + five dev readouts + one reference-Oracle deployment adaptation unless the exact chosen reference was already trained. Component sum, not measured end-to-end. All1026 dev examples charged even for a genre-specific score. Target warmup and omitted process/model transitions remain additional.'))
    baselines=[]
    for family in ['random','genre']:
        for warm in analysis['spec']['heldout_warmups']:
            subset_names=sorted({e['subset'] for e in entries.values() if e.get('warmup_seed')==warm and (e.get('subset','').startswith('candidate_') if family=='random' else e.get('subset','').startswith('genre_'))})
            refs=[reference(next(e for e in entries.values() if e.get('warmup_seed')==warm and e.get('subset')==s)) for s in subset_names]
            baselines.append(dict(family=family,warmup_seed=warm,constant_initial_choice=subset_names[0],constant_initial_choice_seconds=ready(refs[0]),random_expected_training_seconds=float(np.mean([ready(e) for e in refs])),matching_genre_training_seconds={e['subset'][6:]:ready(e) for e in refs} if family=='genre' else None,scope='No candidate simulation or scoring is necessary. Common target warmup excluded. Quality is in paired selection analysis.'))
    direct=[]
    for q in analysis['direct_quality']:
        ee=[e for e in entries.values() if all(e.get(k)==q[k] for k in ['method','kind','steps','lr']) and ('fullpool' if e.get('subset')=='update' else 'random' if e.get('subset','').startswith('candidate_') else 'genre' if e.get('subset','').startswith('genre_') else None)==q['family']]
        assert len(ee)==(3 if q['family']=='fullpool' else 15)
        qualified=[timings[e['tag']] for e in ee if e['tag'] in timings and timings[e['tag']]['three_repeats_complete']]
        refs=[reference(e) for e in ee];refqualified=[timings[e['tag']] for e in refs if e['tag'] in timings and timings[e['tag']]['three_repeats_complete']]
        names=[n for e in ee for n in model_names(e)];off=offline(names)
        record=dict(**{k:q[k] for k in ['family','method','kind','steps','lr']},accuracy_pct=100*q['accuracy']['mean'],delta_vs_oracle_pp=100*q['delta_vs_oracle_same_candidate']['mean'],delta_vs_oracle_ci95_pp=[100*x for x in q['delta_vs_oracle_same_candidate']['ci95']],historical_mean_preparation_plus_online_seconds=float(np.mean([ready(e) for e in ee])),historical_reference_mean_preparation_plus_online_seconds=float(np.mean([ready(e) for e in refs])),offline=off,qualified_timing_adapters=len(qualified),population_adapters=len(ee),timing_subset_scope=sorted({r['subset'] for r in qualified}),qualified_mean_preparation_plus_online_seconds=float(np.mean([r['preparation_plus_online_seconds_median'] for r in qualified])) if qualified else None,qualified_mean_fresh_script_seconds=float(np.mean([r['script_seconds_median'] for r in qualified])) if qualified else None,profiled_matrix_flops=[dict(warmup_seed=r['warmup_seed'],subset=r['subset'],flops=r['total_matrix_flops']) for r in qualified if r['total_matrix_flops']!=''])
        # Only fullpool has representative exact replays for the entire population.
        record['conditional_amortization']=None
        timing_starts={r['warmup_seed'] for r in qualified}
        record['qualified_all_starts']=timing_starts==set(analysis['spec']['heldout_warmups'])
        if not record['qualified_all_starts']:
            record['qualified_mean_preparation_plus_online_seconds']=None
            record['qualified_mean_fresh_script_seconds']=None
        population='fullpool' if q['family']=='fullpool' else 'candidate'
        record['primary_reference_recipe']=analysis['spec'].get('oracle_reference_by_population',{}).get(population,analysis['spec']['oracle_reference'])
        if q['family']=='fullpool' and len(qualified)==3 and len(refqualified)==3:
            saving=float(np.mean([r['preparation_plus_online_seconds_median'] for r in refqualified]))-record['qualified_mean_preparation_plus_online_seconds'];point_within_target=q['delta_vs_oracle_same_candidate']['mean']>=-.01
            count=math.floor(off['recorded_component_sum_seconds']/saving)+1 if saving>0 and point_within_target else None
            record['conditional_amortization']=dict(reference_recipe=record['primary_reference_recipe'],ready_seconds_saved_per_adaptation=saving,mean_quality_within_practical_one_pp=point_within_target,arithmetic_reuses_to_recover_recorded_preparation=count,scope='Arithmetic component-cost scenario against the specified primary population reference only, conditional on comparable quality and continued transfer. It does not establish an advantage over shorter, matched-subset or other cheaper controls. Three-start paired uncertainty is reported; not a noninferiority or general amortization guarantee. No finite value is claimed if ready time is slower or observed mean misses1pp target.')
        direct.append(record)
    for baseline in baselines:
        if baseline['family']!='random':continue
        selector=analysis['spec'].get('random_family_train_all_control')
        if not selector:continue
        matches=[e for e in entries.values() if e.get('warmup_seed')==baseline['warmup_seed'] and all(e.get(k)==v for k,v in selector.items())];assert len(matches)==1;e=matches[0];t=timings.get(e['tag'])
        baseline['train_all_original_pool']=dict(tag=e['tag'],examples=4097,historical_preparation_plus_online_seconds=ready(e),qualified_preparation_plus_online_seconds=t['preparation_plus_online_seconds_median'] if t and t['three_repeats_complete'] else None,qualified_script_seconds=t['script_seconds_median'] if t and t['three_repeats_complete'] else None,scope='No selection needed; permits the whole available original pool rather than a fixed1024-example subset. Not a genre-union control.')
    grouped=defaultdict(list)
    for r in selection:grouped[json.dumps({k:r[k] for k in ['family','method','kind','steps','lr','target','selector']},sort_keys=True)].append(r)
    aggregates=[]
    for signature,rr in sorted(grouped.items()):
        assert len(rr)==3
        aggregates.append(dict(**json.loads(signature),mean_ready_component_sum_seconds=float(np.mean([r['ready_component_sum_seconds'] for r in rr])),mean_candidate_training_component_seconds=float(np.mean([r['candidate_training_component_seconds'] for r in rr])),mean_selection_readout_component_seconds=float(np.mean([r['selection_readout']['total_seconds'] for r in rr])),mean_final_reference_training_component_seconds=float(np.mean([r['final_reference_training_component_seconds'] for r in rr])),recorded_offline_component_sum_seconds=rr[0]['offline']['recorded_component_sum_seconds']))
    result=dict(complete=True,freeze_sha256=sha(FREEZE),decisions_sha256=sha(DECISIONS),policy_sha256=sha(EXP/'configs/surrogate_cost_policy.json'),preparation_catalog_sha256=sha(EXP/'results/surrogate_preparation_components.json'),timing_summary_sha256=sha(timing_path) if timing_path.exists() else None,direct_training=direct,selection_aggregates=aggregates,selection_per_start=selection,cheap_selection_baselines=baselines,unmeasured='Whole deployment-pipeline wall time, offline/readout FLOPs, and repeated clean readout latency are not measured here. Training speed claims require qualified standalone replays; selection costs are explicitly component-sum estimates. Offline historical timer boundaries omit some startup/I/O. Common target warmup is excluded from incremental comparisons.',source_sha256=source_signature(__file__))
    result['shared_target_warmups']=catalog.get('target_warmups',{})
    workflow_path=EXP/'results/selection_pipeline_summary.json'
    result['measured_selection_pipeline']=json.loads(workflow_path.read_text()) if workflow_path.exists() else None
    if result['measured_selection_pipeline'] and result['measured_selection_pipeline']['complete']:
        result['unmeasured']='Whole-workflow timing is measured only for the registered random-family/start302 cases; other selection costs remain component sums. Offline/readout FLOPs are unavailable. Historical offline timer boundaries omit some startup/I/O. Common target warmup is excluded from incremental comparisons.'
    write(EXP/'results/surrogate_cost_comparison.json',result)
    with (EXP/'results/surrogate_cost_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0]),lineterminator='\n');w.writeheader();w.writerows(aggregates)
    log(dict(direct_rows=len(direct),selection_rows=len(selection),aggregate_rows=len(aggregates)))

if __name__=='__main__':main()
