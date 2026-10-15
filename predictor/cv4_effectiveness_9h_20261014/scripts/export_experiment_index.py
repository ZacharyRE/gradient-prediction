"""Export complete recipe settings and development readouts, including negative results."""
from common import *
import csv,collections

def family(name,job):
    if name.startswith('tquery_'):return 'Bernoulli true-gradient queries and matched controls'
    if name.startswith('tcache_'):return 'Frozen teacher-prefix cache (shared optimization)'
    if name.startswith('moments_'):return 'Delayed covariance/variance coefficient calibration'
    if 'multistate' in name or 'ms_' in name or name.startswith('multi_'):return 'Offline multi-state surrogate'
    if 'cachedhead' in name or name.startswith('headcache'):return 'Cached predictor body / head calibration'
    if name.startswith('smooth'):return 'Temporal residual smoothing and matched controls'
    if name.startswith('strat') or 'uniform_reconstruct' in name or name.startswith('recon_uniform'):return 'Stratified sampling and reconstruction controls'
    if name.startswith('memory'):return 'Per-example gradient memory'
    if 'rls' in name:return 'Online recursive least squares'
    if name.startswith('state_'):return 'Explicit LoRA-state conditioning'
    if 'anchor' in name or name.startswith('hold') or '_hold' in name:return 'Periodic gradient anchors and no-predictor hold'
    if job.get('predictor_path'):
        if 'refit_h_' in str(job['predictor_path']) or name.startswith('h_'):return 'Residual hidden-state predictor refit'
        return 'Offline predictor refit'
    if job.get('cal_every') or job.get('cal_on_refresh'):return 'Online neural predictor calibration'
    if job.get('refresh_every'):return 'Periodic full-batch true-gradient refresh'
    if job.get('lambda_mode') or job.get('coefficient',1)!=1:return 'Control-variate coefficient tuning'
    if name.startswith('long'):return 'Long-horizon Oracle/CV curves'
    if name.startswith('gpu_'):return 'GPU-resident features (shared optimization)'
    return 'Oracle/CV baselines and learning-rate controls'

def main():
    from export_preparation_costs import main as export_preparation
    export_preparation()
    preparation=json.loads((EXP/'results/predictor_preparation_costs.json').read_text())
    costs={r['predictor_sha256']:r['total_recorded_preparation_seconds'] for r in preparation['rows']}
    dev=list(csv.DictReader((EXP/'results/development_live.csv').open()))
    by_tag={r['tag']:r for r in dev} if dev and 'tag' in dev[0] else {}
    holder=json.loads((EXP/'results/gpu_reservation.json').read_text())
    known={holder['pid']}|{int(p.stem) for p in (EXP/'results/worker_memory_reservations').glob('*.json')}
    observations=read(EXP/'results/gpu_observations.jsonl')
    foreign_times=[o['time'] for o in observations if any(int(line.split(',')[1]) not in known for line in o['processes'])]
    rows=[]
    for path in sorted((EXP/'models').glob('*/summary.json')):
        config_path=path.parent/'config.json'
        if not config_path.exists():continue
        config=json.loads(config_path.read_text());job=config.get('job',{});name=job.get('name',path.parent.name)
        if name.startswith(('profile__','timing__')):continue
        summary=json.loads(path.read_text())
        if not summary.get('entries'):continue
        history_path=path.parent/'history.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else []
        cumulative={};true_count=0;cal_seconds=0.;queries=0;anchors=0
        measured_components={k:all(k in h for h in history) for k in ['prefix_prediction_seconds','label_seconds']}
        component_seconds={k:0. for k in measured_components}
        for h in history:
            true_count+=h.get('true_gradient_examples',0);cal_seconds+=h.get('calibration_seconds',0.)
            queries+=int(h.get('queried',False));anchors+=int(h.get('anchor',False))
            cumulative[h['step']]=dict(true_gradient_examples_through_checkpoint=true_count,calibration_seconds_through_checkpoint=cal_seconds,query_calls_through_checkpoint=queries,anchor_calls_through_checkpoint=anchors)
            for key,measured in measured_components.items():
                component_seconds[key]+=h.get(key,0.)
                cumulative[h['step']][key+'_through_checkpoint']=component_seconds[key] if measured else ''
        calibration_full=summary.get('calibration_seconds')
        if calibration_full is None:
            calibration_full=sum(h.get('calibration_seconds',0) for h in history)
        overlap=any(config_path.stat().st_mtime-15<=t<=path.stat().st_mtime+15 for t in foreign_times)
        for e in summary.get('entries',[]):
            matches=[r for r in dev if r['task']==e['task'] and r['method']==e['method'] and int(r['seed'])==e['seed'] and int(r['steps'])==e['steps']]
            assert len(matches)<=1,(path,e['tag'])
            rows.append(dict(task=e['task'],family=family(name,job),method=name,update_seed=e['seed'],predictor_seed=e.get('predictor_seed'),steps=e['steps'],dev_accuracy_pct=matches[0]['dev_accuracy_pct'] if matches else '',online_seconds=e['online_seconds'],preparation_seconds=e.get('cache_seconds',0),offline_initialization_seconds=e.get('offline_fit_seconds',summary.get('offline_fit_seconds',0)),offline_gradient_examples=e.get('offline_gradient_examples',summary.get('offline_gradient_examples',0)),gradient_memory_parameters=config.get('gradient_memory_parameters',0),online_occupancy_observation='foreign_context_overlap_requires_replay' if overlap else 'sampled_owned_only_during_config_to_summary_interval',calibration_seconds_full_run=calibration_full,true_gradient_examples_full_run=summary.get('true_gradient_examples'),job_json=json.dumps(job,sort_keys=True),config_path=str(config_path.relative_to(EXP)),adapter_sha256=e['adapter_sha256']))
            predictor_sha=config.get('predictor_sha256')
            rows[-1]['inherited_predictor_preparation_seconds']=costs.get(predictor_sha,'unknown') if predictor_sha else 0.
            rows[-1]['reused_label_subset_collection_seconds']='not_separately_measured' if job.get('offline_init') else 0.
            rows[-1].update(cumulative.get(e['steps'],dict(true_gradient_examples_through_checkpoint='',calibration_seconds_through_checkpoint='',query_calls_through_checkpoint='',anchor_calls_through_checkpoint='',prefix_prediction_seconds_through_checkpoint='',label_seconds_through_checkpoint='')))
            rows[-1].update(code_sha256=config.get('code_sha256',''),source_bundle_sha256=config.get('source_bundle_sha256',''),predictor_sha256=predictor_sha or '',warmup_sha256=config.get('warmup_sha256',''),update_data_sha256=config.get('update_sha256',''))
    with (EXP/'results/experiment_index.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    groups=collections.defaultdict(list)
    for r in rows:
        if r['dev_accuracy_pct']!='':groups[r['task'],r['family']].append(r)
    output=[]
    for (task,kind),items in sorted(groups.items()):
        best=max(items,key=lambda r:float(r['dev_accuracy_pct']))
        output.append(dict(task=task,family=kind,trajectories=len({(r['method'],r['update_seed'],r['predictor_seed']) for r in items}),readouts=len(items),best_observed_development_accuracy_pct=float(best['dev_accuracy_pct']),best_method=best['method'],best_step=best['steps'],best_seed=best['update_seed']))
    write(EXP/'results/family_screening_summary.json',dict(selection_only=True,note='Best observed single development readout is optimistic after screening; it is not independent validation or a multi-seed average. All configurations and negative results remain in experiment_index.csv. Timing occupancy is audited separately.',rows=output))
    log(dict(checkpoint_rows=len(rows),families=len(groups)))

if __name__=='__main__':main()
