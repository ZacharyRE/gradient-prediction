"""Copy this study's reviewable evidence into the isolated publication worktree.

No commit/push and no changes to the live experiment or unrelated source worktree.
Raw prompts/responses, model tensors and operational process logs stay local.
"""
from common import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worktree',type=Path,required=True);a=ap.parse_args()
    dest=a.worktree.resolve();assert (dest/'.git').exists() and dest!=ROOT
    report=dest/'reports/predictor_surrogate_20261015';code=dest/'predictor'/EXP.name;copied=[]
    def public_provenance(value):
        if isinstance(value,list):return [public_provenance(v) for v in value]
        if not isinstance(value,dict):return value
        return {k:public_provenance(v) for k,v in value.items()
                if k!='pid' and not k.endswith('_pid') and k not in
                {'process_start','uid','cmdline','cmdline_sha256','foreign_processes'}}
    def copy(source,target):
        source=Path(source)
        if not source.is_file():return
        target.parent.mkdir(parents=True,exist_ok=True)
        transformed=False
        if source.parent==EXP/'provenance' and source.suffix=='.json':
            original=json.loads(source.read_text());public=public_provenance(original)
            if public!=original:
                public['_publication_redaction']=dict(original_sha256=sha(source),scope='Operational process identifiers omitted; scientific registration fields retained.')
                write(target,public);transformed=True
        if not transformed:shutil.copy2(source,target)
        row=dict(source=str(source.relative_to(EXP)),destination=str(target.relative_to(dest)),sha256=sha(target))
        if transformed:row.update(source_sha256=sha(source),transformation='Remove operational process identifiers from public provenance')
        copied.append(row)
    for source in sorted((EXP/'scripts').glob('*.py')):copy(source,code/'scripts'/source.name)
    for source in sorted((EXP/'report').glob('*.md')):copy(source,report/source.name)
    copy(EXP/'literature/NOTES.md',report/'LITERATURE.md')
    # Include scientific recipes, not local GPU admission/process policies.
    config_names=['protocol.json','development_selection_rule.json','secondary_analysis_plan.json','surrogate_final_freeze.json','surrogate_selection_decisions.json','surrogate_analysis_spec.json','surrogate_final_recipe_plan.json','surrogate_final_timing_plan.json','run_deadline.json']
    config_names+=['behavior_common_selection_plan.json','queue_surrogate_final_readouts.json']
    config_names+=['surrogate_cost_policy.json','queue_surrogate_final_timing.json','queue_selection_pipeline.json']
    config_names+=['surrogate_supplemental_timing_plan.json','queue_surrogate_supplemental_timing.json']
    config_names+=['optional_fit_completion_policy.json','diversity_states.json','queue_diversity2048.json','queue_late_kd.json','queue_capacity512.json']
    config_names+=['kd_readout_completion_policy.json','queue_distill_priority.json']
    config_names+=['registered_controls_completion_policy.json']
    config_names+=['token_behavior_readout_plan.json','token_behavior_entries.json']
    for source in sorted((EXP/'configs').glob('*_jobs.json')):copy(source,code/'configs'/source.name)
    controls=json.loads((EXP/'configs/registered_controls_completion_policy.json').read_text())
    for stage in controls['stages'].values():
        argv=stage['training_argv'];source=Path(argv[argv.index('--jobs')+1])
        assert source.parent==EXP/'configs'
        if not any(r['source']==str(source.relative_to(EXP)) for r in copied):
            copy(source,code/'configs'/source.name)
    for name in config_names:copy(EXP/'configs'/name,code/'configs'/name)
    for name in ['critic_states.json','critic_fast_states.json','rollout_shift_states.json','data_manifest.json','genre_candidates.json','factor_scale_selection.json','blind_matched_plan.json']:
        copy(EXP/'configs'/name,code/'configs'/name)
    for source in sorted((EXP/'configs').glob('surrogate_final_entries*.json')):copy(source,code/'configs'/source.name)
    result_names=['reuse_experiment_index.csv','reuse_experiment_inventory.json','development_summary.csv','candidate_confirmation_summary.json','candidate_confirmation_summary.csv','development_three_start_controls.json','development_three_start_paired.json','surrogate_preparation_components.json','standalone_reuse_timing_summary.json','standalone_reuse_timing_summary.csv','final_selection_analysis.json','final_selection_analysis.csv','final_selection_predictions.npz','final_selection_prediction_bundle_audit.json','surrogate_cost_comparison.json','surrogate_cost_comparison.csv','cv_variance_audit.json','batch_mean_error_cpu_audit.json','graph_rollout_audit.json','graph_labels_audit.json','graph_flop_accounting_audit.json','data_reproduction_audit.json','state_screen_scope.json']
    result_names+=['frozen_correctness.npz','frozen_correctness_audit.json','fullpool_confirmation_summary.json','fullpool_confirmation_summary.csv','development_reuse_frontier.json','development_equal_teacher_budget.json','predictor_rollout_shift.json','direct_quality_population_cpu_audit.json','state321_readout_replay_audit.json']
    result_names+=['behavior_development_summary.json','anchored_behavior_cpu_audit.json','early_predictor_cutoff_audit.json']
    result_names+=['backend_confirmation_summary.json','backend_confirmation_summary.csv']
    result_names+=['position_candidate_confirmation_summary.json','position_candidate_confirmation_summary.csv']
    result_names+=['coverage_rollout_shift.json','coverage_fit_budget_audit.json']
    result_names+=['scalar_cv_diagnostic.json','scalar_cv_diagnostic.csv']
    result_names+=['selected_predictor_architectures.json']
    result_names+=['resource_release.json']
    result_names+=['late_development_summary.json','late_development_summary.csv']
    result_names+=['late_confirmation_gate_decision.json']
    result_names+=['kd_readout_completion.json']
    result_names+=['optional_completed_readout_inventory.json']
    result_names+=['registered_controls_completion.json']
    result_names+=['token_behavior_input_audit.json','token_behavior_development_summary.json']
    result_names+=['late_generated_records_audit.json']
    result_names+=['development_cutoff_receipt.json']
    result_names+=['selection_analysis_cpu_audit.json','paired_controls_development_cpu_audit.json','selection_pipeline_summary.json','selection_pipeline_summary.csv','selection_pipeline_queue_receipt.json','study_execution_summary.json','final_behavior_fidelity.json','final_behavior_fidelity.csv','surrogate_analysis_reproduction_audit.json','behavior_common_selection.json','surrogate_readout_queue_receipt.json','surrogate_timing_queue_receipt.json','surrogate_entry_memberships.json','surrogate_entry_assembly_status.json','surrogate_cpu_audits_complete.json','split_integrity_audit.json','final_evidence_audit.json','generated_records_audit.json']
    for name in result_names:copy(EXP/'results'/name,report/'results'/name)
    for folder in ['figures','reuse_timing','reuse_flops','backend_equivalence','selection_pipeline','pipeline_training']:
        for source in sorted((EXP/'results'/folder).glob('*')):
            if source.suffix in ['.png','.pdf','.json']:copy(source,report/'results'/folder/source.name)
    for source in sorted((EXP/'models').glob('mnli_*/summary.json')):
        if '__w' in source.parent.name or not (source.parent/'best.pt').exists():continue
        for name in ['summary.json','config.json','history.json']:
            copy(source.parent/name,report/'results/predictor_fit_receipts'/source.parent.name/name)
    for w in [302,303,304]:
        source_root=EXP/'models'/f'mnli_warmup_w{w}'
        for name in ['summary.json','adapter_config.json']:copy(source_root/name,report/'results/target_warmup_receipts'/str(w)/name)
    for folder in ['eval','behavior']:
        for protocol in sorted((EXP/'results'/folder).glob('*/protocol.json')):
            d=json.loads(protocol.read_text())
            if d['args']['split'] not in ['selection_dev','selection_holdout']:continue
            for name in ['protocol.json','summary.json','runtime_receipt.json','complete.json']:
                copy(protocol.parent/name,report/'results'/folder/protocol.parent.name/name)
    for name in ['protocol.json','summary.json','complete.json']:
        copy(EXP/'results/behavior/token_behavior_dev'/name,
             report/'results/behavior/token_behavior_dev'/name)
    for name in ['environment.json','data_reproduction.json.gz','data_reproduction_audit.json','code_validation.json','reproduction_smoke.json','user_one_hour_extension.json','data_download_source_verified.json','compute_cutoff_admission_fix.json','workflow_reference_fallback_fix.json','publication_layout_statistics_reproduction.json']:
        copy(EXP/'provenance'/name,report/'provenance'/name)
    for source in sorted((EXP/'provenance').glob('*registration.json')):copy(source,report/'provenance'/source.name)
    copy(EXP/'provenance/overview_render.json',report/'provenance/overview_render.json')
    copy(EXP/'provenance/optional_fit_budget_cpu_audit.json',report/'provenance/optional_fit_budget_cpu_audit.json')
    copy(EXP/'provenance/optional_fit_validation_audit.json',report/'provenance/optional_fit_validation_audit.json')
    copy(EXP/'provenance/diversity_existing_control_replay.json',report/'provenance/diversity_existing_control_replay.json')
    copy(EXP/'provenance/late_confirmation_development_snapshot.json',report/'provenance/late_confirmation_development_snapshot.json')
    copy(EXP/'provenance/kd_readout_freeze.json',report/'provenance/kd_readout_freeze.json')
    copy(EXP/'provenance/kd_readout_budget_cpu_audit.json',report/'provenance/kd_readout_budget_cpu_audit.json')
    copy(EXP/'provenance/late_rollout_identity_audit.json',report/'provenance/late_rollout_identity_audit.json')
    copy(EXP/'provenance/diversity_cached_label_identity.json',report/'provenance/diversity_cached_label_identity.json')
    copy(EXP/'provenance/registered_controls_budget_cpu_audit.json',report/'provenance/registered_controls_budget_cpu_audit.json')
    copy(EXP/'provenance/token_behavior_predictor_freeze.json',report/'provenance/token_behavior_predictor_freeze.json')
    for name in ['state_evaluation_reprioritization.json','graph_label_audit_storage_cast.json','anchored_initial_identity_fix.json']:
        copy(EXP/'provenance'/name,report/'provenance'/name)
    selected=[];freeze=EXP/'configs/surrogate_final_freeze.json'
    if freeze.exists():
        for record in json.loads(freeze.read_text())['entries'].values():
            e=record['entry'];root=Path(e['adapter']).parent
            if e.get('steps')==0:continue
            cfg=json.loads((root/'config.json').read_text());history=json.loads((root/'history.json').read_text())
            checkpoint=next(r for r in history if r['step']==e['steps'])
            selected.append(dict(entry=e,config=cfg,checkpoint_diagnostic=checkpoint,history_sha256=sha(root/'history.json'),training_summary_sha256=sha(root/'summary.json')))
        target=report/'results/selected_training_receipts.json';write(target,dict(rows=selected))
        copied.append(dict(source='generated selected training receipts',destination=str(target.relative_to(dest)),sha256=sha(target)))
    write(report/'provenance/publication_manifest.json',dict(created_epoch=time.time(),files=copied,exclusions='Model tensors, raw prompts/generated responses and operational resource/process logs remain local. Reproduction metadata reconstructs all full inputs from the specified original MNLI parquet. Compact correctness bundles support aggregate statistical reproduction, not independent response regrading. Historical absolute paths identify receipts and require relocation for rerunning. Current scripts are supplied; immutable per-stage snapshots and their hashes remain in local provenance.'))
    log(dict(worktree=str(dest),report=str(report),copied_files=len(copied),selected_receipts=len(selected)))

if __name__=='__main__':main()
