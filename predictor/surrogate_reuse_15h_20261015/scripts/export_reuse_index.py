"""Export every completed rollout recipe; preserve negative development results."""
from common import *
import csv

def family(job):
    kind=job['kind'];path=job.get('predictor_path','');name=job['name']
    if kind in ['full_table','table']:return 'full_dataset_gradient_table' if kind=='full_table' else 'minibatch_gradient_table'
    if 'tangent' in name or kind.startswith(('tangent','linear')) or kind in ['oracle_class','oracle_full']:return 'local_suffix_linearization_and_objective_controls'
    if job.get('burnin_steps'):return 'true_gradient_burnin'
    if job.get('cal_every') or job.get('cal_on_refresh'):return 'sparse_online_calibration'
    for token,label in [('position_','position_gradient_template'),('factor_scale','factor_magnitude_regression'),('factor_','class_covariance_factor'),('copied','copied_teacher_suffix'),('state','explicit_lora_state'),('token','causal_token_critic'),('sobolev','scalar_gradient_critic'),('value','scalar_value_critic'),('vector','direct_gradient_predictor')]:
        if token in path:return label
    if kind in ['transport','class_ema','class_cov','stratified']:return kind
    if kind in ['neural_hybrid','held_gradient','zero_gradient']:return kind
    return 'original_predictor_and_true_gradient_controls'

def main():
    from summarize_reuse import rows as development
    from export_surrogate_preparation import main as preparation
    preparation();costs=json.loads((EXP/'results/surrogate_preparation_components.json').read_text());costs={r['predictor_sha256']:r for r in costs['predictors']}
    by_tag={}
    for r in development:
        if r['tag'] in by_tag:assert by_tag[r['tag']]['accuracy']==r['accuracy']
        by_tag[r['tag']]=r
    rows=[];trajectories=0
    for path in sorted((EXP/'models').glob('*/summary.json')):
        config_path=path.parent/'config.json'
        if not config_path.exists():continue
        summary=json.loads(path.read_text());cfg=json.loads(config_path.read_text());job=cfg.get('job')
        if not job or not summary.get('complete') or not summary.get('entries') or job['name'].startswith(('replay_','profile_','timing_')):continue
        trajectories+=1;history_path=path.parent/'history.json';history=json.loads(history_path.read_text());cumulative={};truth=0;cal=0.
        for h in history:
            truth+=h.get('true_gradient_examples',0);cal+=h.get('calibration_seconds',0.);cumulative[h['step']]=(truth,cal)
        for e in summary['entries']:
            d=by_tag.get(e['tag']);psha=cfg.get('predictor_sha256') or cfg.get('online_surrogate_sha256');cost=costs.get(psha)
            count,seconds=cumulative[e['steps']]
            rows.append(dict(tag=e['tag'],task=e['task'],warmup_seed=e.get('warmup_seed'),subset=e.get('subset',cfg.get('subset')),update_seed=e['seed'],family=family(job),method=job['name'],kind=job['kind'],k=job.get('k'),lr=job['lr'],steps=e['steps'],teacher_backend=cfg.get('teacher_backend','native'),cache_policy=cfg.get('cache_policy','all'),development_accuracy_pct=d['accuracy'] if d else '',development_output_cap_hits=d['output_cap_hits'] if d else '',online_seconds=e['online_seconds'],preparation_seconds=e.get('cache_seconds',0.),calibration_seconds=seconds,true_gradient_examples=count,initialization_true_gradient_examples=cfg.get('initialization_true_gradient_examples',summary.get('initialization_true_gradient_examples',0)),predictor_sha256=psha or '',recorded_predictor_preparation_seconds=cost['recorded_preparation_seconds_excluding_source_warmups'] if cost else '' if psha else 0.,recorded_predictor_source_warmup_seconds=cost['recorded_source_warmup_seconds'] if cost else '' if psha else 0.,adapter_sha256=e['adapter_sha256'],source_sha256=cfg.get('source_sha256'),config_sha256=sha(config_path),history_sha256=sha(history_path),job_json=json.dumps(job,sort_keys=True)))
    with (EXP/'results/reuse_experiment_index.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    failures=[]
    for line in read(EXP/'results/stage_attempts.jsonl'):
        if line['returncode']:
            item={k:line[k] for k in ['name','started','finished','returncode','reason']}
            item['classification']='intentional_screening_reprioritization' if line['name']=='pilot_state_eval' and (EXP/'provenance/state_evaluation_reprioritization.json').exists() else 'failed_attempt'
            if line['name']=='state_candidate_w321_eval' and (EXP/'results/state321_readout_replay_audit.json').exists():item['classification']='owned_vllm_teardown_false_alarm_all_records_cleanly_reproduced'
            cutoff=EXP/'results/development_cutoff_receipt.json'
            if line['name']=='finish_controls_pilot_table' and cutoff.exists():
                receipt=json.loads(cutoff.read_text())
                if receipt['expected_bounded_stop'] and line['finished']==receipt['finished_epoch']:
                    item['classification']='registered_development_budget_cutoff_not_a_quality_failure'
            failures.append(item)
    write(EXP/'results/reuse_experiment_inventory.json',dict(trajectories=trajectories,checkpoint_rows=len(rows),development_readouts=sum(r['development_accuracy_pct']!='' for r in rows),failed_stage_attempts=failures,scope='All completed rollout recipes, excluding numerical/timing/profile replays. Blank development accuracy means not evaluated; heldout-start outcomes are not silently treated as development. Times are within-process historical components, not qualified standalone repeats. Failed attempts remain explicit. All source hash receipts retained locally.'))
    log(dict(trajectories=trajectories,checkpoints=len(rows),failed_attempts=len(failures)))

if __name__=='__main__':main()
