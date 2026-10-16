"""Check complete registered scientific coverage before interpreting final tables.

This verifies experimental artifacts; publication and GPU cleanup remain separate.
"""
from common import *
from surrogate_freeze import FREEZE,DECISIONS,critical_code,entry_key

def main():
    paths={}
    def read_json(relative):
        p=EXP/relative;assert p.is_file(),f'Missing required evidence: {relative}';paths[relative]=sha(p);return json.loads(p.read_text())
    plan=read_json('configs/surrogate_final_recipe_plan.json');frozen=read_json('configs/surrogate_final_freeze.json');decisions=read_json('configs/surrogate_selection_decisions.json');expected=set(frozen['entries']);tags={v['entry']['tag'] for v in frozen['entries'].values()}
    assert frozen['complete'] and len(expected)==plan['expected_unique_entries']
    assert decisions['complete'] and decisions['freeze_sha256']==sha(FREEZE) and decisions['no_holdout_outcomes_used']
    assert frozen['code']==critical_code()
    for path,r in frozen['recipes'].items():assert sha(path)==r['sha256'],path
    for path,want in frozen['behavior_predictors'].items():assert sha(path)==want,path
    gradient_predictors={}
    for key,r in frozen['entries'].items():
        e=r['entry'];assert sha(Path(e['adapter'])/'adapter_model.safetensors')==r['adapter_sha256'],key
        if e.get('predictor_path'):
            cfg=json.loads((Path(e['adapter']).parent/'config.json').read_text());path=e['predictor_path'];expected_predictor=cfg['predictor_sha256'];assert expected_predictor
            if path in gradient_predictors:assert gradient_predictors[path]==expected_predictor
            else:assert sha(path)==expected_predictor;gradient_predictors[path]=expected_predictor

    metadata=read_json('results/selected_predictor_architectures.json')
    assert metadata['complete'] and metadata['freeze_sha256']==sha(FREEZE)
    expected_predictor_hashes=set(gradient_predictors.values())|set(frozen['behavior_predictors'].values())
    assert {r['checkpoint_sha256'] for r in metadata['models']}==expected_predictor_hashes
    assert all(r['strict_state_load_passed'] and r['parameter_count']>0 for r in metadata['models'])
    coverage={}
    for split in ['selection_dev','selection_holdout']:
        n=len(read(data('mnli',split)))
        root=f'results/eval/surrogate_final_{split}';protocol=read_json(root+'/protocol.json');summary=read_json(root+'/summary.json')
        assert protocol['surrogate_final_freeze']['freeze_sha256']==sha(FREEZE)
        assert protocol['full_inputs'] and protocol['input_token_ids_verified_at_generation'] and protocol['max_new_tokens']==256
        assert protocol['max_prompt_tokens']+protocol['max_new_tokens']<=protocol['context']
        assert set(summary)==expected and all(r['n']==n for r in summary.values())
        if split=='selection_holdout':assert protocol['surrogate_final_freeze']['decisions_sha256']==sha(DECISIONS)
        root=f'results/behavior/surrogate_final_behavior_{split}';complete=read_json(root+'/complete.json');summary=read_json(root+'/summary.json');protocol=read_json(root+'/protocol.json')
        assert complete['complete'] and complete['entries']==len(expected) and set(summary)==tags
        assert protocol['full_original_prompts'] and protocol['gold_class_never_enters_model_input'] and protocol['freeze']['freeze_sha256']==sha(FREEZE)
        coverage[split]=dict(generated_entries=len(expected),behavior_entries=len(tags),examples=n)
    analysis=read_json('results/final_selection_analysis.json');assert analysis['complete'] and analysis['freeze_sha256']==sha(FREEZE) and analysis['decisions_sha256']==sha(DECISIONS)
    bundle=read_json('results/frozen_correctness_audit.json');assert bundle['passed'] and bundle['entries']==len(expected) and bundle['sha256']==sha(EXP/'results/frozen_correctness.npz')
    reproduction=read_json('results/surrogate_analysis_reproduction_audit.json');assert reproduction['passed'] and reproduction['source_result_sha256']==sha(EXP/'results/final_selection_analysis.json') and reproduction['bundle_sha256']==bundle['sha256']
    fidelity=read_json('results/final_behavior_fidelity.json');assert fidelity['complete'] and fidelity['freeze_sha256']==sha(FREEZE) and all(r['logits_exact_initial'] for r in fidelity['warmup_identity_checks'])
    split_audit=read_json('results/split_integrity_audit.json');assert split_audit['passed'] and split_audit['freeze_sha256']==sha(FREEZE)
    generated=read_json('results/generated_records_audit.json');assert generated['passed'] and generated['freeze_sha256']==sha(FREEZE) and generated['decisions_sha256']==sha(DECISIONS)
    assert all(v['complete_source_prompts_verified'] and v['grading_fields_recomputed'] and len(v['entries'])==len(expected) for v in generated['splits'].values())
    tp=read_json('results/surrogate_timing_queue_receipt.json');supplement=read_json('configs/surrogate_supplemental_timing_plan.json');timing=read_json('results/standalone_reuse_timing_summary.json');bytag={r['tag']:r for r in timing['rows']};reps=plan['timing_plan']['repetitions'];profiled=0
    assert supplement['freeze_sha256']==sha(FREEZE) and supplement['registered_after_quality_readouts']
    assert supplement['repetitions']==reps and supplement['flops_starts']==plan['timing_plan']['flops_starts']
    timing_references=tp['reference_entries']+supplement['reference_entries'];assert len({e['tag'] for e in timing_references})==len(timing_references)
    frozen_bytag={v['entry']['tag']:v['entry'] for v in frozen['entries'].values()}
    for e in timing_references:
        assert frozen_bytag[e['tag']]==e
        r=bytag[e['tag']];assert r['qualified_repetitions']==reps and r['three_repeats_complete'] and r['weight_sha256']==e['adapter_sha256']
        for rec in r['selected_timing_receipts']:
            assert sha(rec['path'])==rec['sha256'];actual=json.loads(Path(rec['path']).read_text());assert actual['weights_bitwise_equal'] and actual['timing_valid_for_reference'] and actual['reference_entry']==e
        if e['warmup_seed'] in plan['timing_plan']['flops_starts']:assert r['total_matrix_flops']>0;profiled+=1
    assert profiled==tp['flop_stages']+supplement['flop_stages']
    pipeline=read_json('results/selection_pipeline_summary.json');cost_policy=read_json('configs/surrogate_cost_policy.json')['measured_selection_pipeline_plan'];assert pipeline['complete'] and len(pipeline['rows'])==len(cost_policy['families'])*len(cost_policy['conditions'])
    for row in pipeline['rows']:
        assert row['three_repeats_complete'] and row['qualified_repetitions']==cost_policy['repetitions']
        for rec in row['receipts']:
            assert sha(rec['path'])==rec['sha256'];r=json.loads(Path(rec['path']).read_text());assert r['fresh_predictions_match'] and r['selected_candidate_matches'] and r['freeze_sha256']==sha(FREEZE) and r['decisions_sha256']==sha(DECISIONS)
            for training in r['training_receipts']:assert sha(training['path'])==training['sha256'] and training['weights_bitwise_equal']
    cost=read_json('results/surrogate_cost_comparison.json')
    assert cost['complete'] and cost['freeze_sha256']==sha(FREEZE) and cost['decisions_sha256']==sha(DECISIONS)
    assert cost['preparation_catalog_sha256']==sha(EXP/'results/surrogate_preparation_components.json')
    assert cost['timing_summary_sha256']==sha(EXP/'results/standalone_reuse_timing_summary.json')
    assert cost['measured_selection_pipeline']==pipeline
    bytag={v['entry']['tag']:v['entry'] for v in frozen['entries'].values()}
    fields=['family','warmup_seed','method','kind','steps','lr','target','selector']
    ledger={tuple(r[k] for k in fields):r for r in cost['selection_per_start']}
    assert len(ledger)==len(decisions['decisions'])
    reference_selector=analysis['spec'].get('oracle_reference_by_population',{}).get('candidate',analysis['spec']['oracle_reference'])
    for decision in decisions['decisions']:
        row=ledger[tuple(decision[k] for k in fields)];assert row['selected_subset']==decision['selected_subset']
        refs=[e for e in bytag.values() if e.get('warmup_seed')==decision['warmup_seed'] and e.get('subset')==decision['selected_subset'] and all(e.get(k)==v for k,v in reference_selector.items())]
        assert len(refs)==1;ref=refs[0];already=ref['tag'] in {e['tag'] for e in decision['candidate_scores']}
        assert row['chosen_reference_already_trained']==already
        expected_final=0. if already else ref.get('cache_seconds',0)+ref['online_seconds']
        assert abs(row['final_reference_training_component_seconds']-expected_final)<1e-9
    result=dict(passed=True,epoch=time.time(),freeze_sha256=sha(FREEZE),decision_sha256=sha(DECISIONS),gradient_predictor_sha256=gradient_predictors,readout_coverage=coverage,timing_references=len(timing_references),primary_timing_references=len(tp['reference_entries']),supplemental_timing_references=len(supplement['reference_entries']),supplement_registered_after_quality_readouts=True,timing_repetitions=reps,flop_profiles=profiled,workflow_conditions=len(pipeline['rows']),workflow_repetitions=cost_policy['repetitions'],evidence_sha256=paths,scope='Registered experiment coverage, artifact identities, full-input protocols, separate dev/holdout decisions, statistical reproduction, qualified timing and actual workflow receipts. The timing-only supplement covers all previously omitted frozen fullpool conditions and was registered after quality readouts. This is not by itself scientific proof of superiority, a publication audit, or a resource cleanup receipt.')
    write(EXP/'results/final_evidence_audit.json',result);log({k:result[k] for k in ['passed','timing_references','timing_repetitions','flop_profiles','workflow_conditions']})

if __name__=='__main__':main()
