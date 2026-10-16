"""Copy reviewable code and compact evidence into an isolated Git worktree.

Does not commit or push. Excludes datasets, weights, raw predictions, operational
process logs, and foreign-process details. Existing unrelated files are untouched.
"""
from common import *
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worktree',type=Path,required=True);a=ap.parse_args()
    dest=a.worktree.resolve();assert (dest/'.git').exists() and dest!=ROOT
    report=dest/'reports/predictor_efficiency_20261014'
    code=dest/'predictor'/EXP.name
    copied=[]
    def copy(source,target):
        source=Path(source)
        if not source.is_file():return
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        copied.append(dict(source=str(source.relative_to(EXP)),destination=str(target.relative_to(dest)),sha256=sha(source)))
    for source in (EXP/'scripts').glob('*.py'):copy(source,code/'scripts'/source.name)
    for source in (EXP/'report').glob('*.md'):copy(source,report/source.name)
    for source in (EXP/'configs').glob('*_jobs.json'):copy(source,code/'configs'/source.name)
    for name in ['protocol.json','final_freeze.json','final_timing_policy.json','final_timing_plan.json','final_selection_plan.json','paws_frozen_entries.json','mnli_frozen_entries.json']:
        copy(EXP/'configs'/name,code/'configs'/name)
    names=['experiment_index.csv','development_cohorts.json','family_screening_summary.json','long_curves.csv','predictor_preparation_costs.json','frozen_test_analysis.json','frozen_oracle_confirmation_analysis.json','frozen_timing_analysis.json','time_query_stability.json','teacher_cache_trajectory_comparison.json','teacher_prefix_audit.json','numerical_audit.json','gpu_feature_audit.json','fast_backend_audit.json','ridge_numerical_audit.json','cv_variance_audit.json','time_query_algebra_audit.json','stratified_estimator_audit.json','flop_counter_cpu_audit.json','all_split_metadata_audit.json']
    for name in names:copy(EXP/'results'/name,report/'results'/name)
    for name in ['final_comparison.csv','final_comparison.json','development_frontier.json','h_cache_audit.json','cached_head_audit.json','state_feature_audit.json','surrogate_transfer.csv','teacher_cache_long_comparison.json','timing_occupancy_audit.json']:copy(EXP/'results'/name,report/'results'/name)
    for source in (EXP/'results').glob('surrogate_transfer_*.json'):copy(source,report/'results'/source.name)
    for folder in ['figures','flops','timing']:
        for source in (EXP/'results'/folder).glob('*'):
            if source.suffix in ['.png','.pdf','.json']:copy(source,report/'results'/folder/source.name)
    for task in ['paws','mnli']:
        for split in ['confirmation','test']:
            folder=f'{task}_frozen_{split}'
            for name in ['protocol.json','summary.json']:
                copy(EXP/'results/eval'/folder/name,report/'results/eval'/folder/name)
    copy(EXP/'results/frozen_control_comparisons.json',report/'results/frozen_control_comparisons.json')
    copy(EXP/'results/supplementary_cost_execution.json',report/'results/supplementary_cost_execution.json')
    for name in ['environment.json','replay_environment.json','code_validation.json','reproduction_smoke.json']:
        copy(EXP/'provenance'/name,report/'provenance'/name)
    for source in (EXP/'provenance').glob('*registration.json'):
        copy(source,report/'provenance'/source.name)
    selected=[]
    freeze=EXP/'configs/final_freeze.json'
    if freeze.exists():
        for entry in json.loads(freeze.read_text())['entries']:
            if entry.get('cohort')=='warmup':continue
            root=Path(entry['adapter']).parent
            config=json.loads((root/'config.json').read_text())
            history=json.loads((root/'history.json').read_text())
            checkpoint=next(r for r in history if r['step']==entry['steps'])
            selected.append(dict(entry=entry,config=config,checkpoint_diagnostic=checkpoint,training_summary_sha256=sha(root/'summary.json'),history_sha256=sha(root/'history.json')))
        target=report/'results/selected_training_receipts.json';write(target,dict(rows=selected))
        copied.append(dict(source='generated selected training receipts',destination=str(target.relative_to(dest)),sha256=sha(target)))
    write(report/'provenance/publication_manifest.json',dict(created_epoch=time.time(),files=copied,excluded='Datasets, model weights, raw generated responses, process identities and reservation/incident logs. These remain local. Training recipes and source code are included; reproduction requires the recorded data/warmup/predictor artifacts.'))
    log(dict(worktree=str(dest),copied_files=len(copied),selected_receipts=len(selected)))


if __name__=='__main__':main()
