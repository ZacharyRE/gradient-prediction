"""Summarize clean standalone replays of frozen adapters, retaining mismatches."""
from common import *
import collections


def stats(values):
    return dict(median=float(np.median(values)),minimum=float(min(values)),maximum=float(max(values)))


def main():
    freeze=json.loads((EXP/'configs/final_freeze.json').read_text())
    allowed={(e['task'],e['method'],e['steps'],e['adapter_sha256']):e for e in freeze['entries']}
    costs={r['predictor_sha256']:r['total_recorded_preparation_seconds'] for r in json.loads((EXP/'results/predictor_preparation_costs.json').read_text())['rows']}
    holder=json.loads((EXP/'results/gpu_reservation.json').read_text())
    owned={holder['pid']}|{int(p.stem) for p in (EXP/'results/worker_memory_reservations').glob('*.json')}
    observations=read(EXP/'results/gpu_observations.jsonl')
    foreign=[r['time'] for r in observations if any(int(line.split(',')[1]) not in owned for line in r['processes'])]
    groups=collections.defaultdict(list);rejected=[]
    for path in sorted((EXP/'results/timing').glob('*.json')):
        r=json.loads(path.read_text());key=(r['task'],r['recipe'],r['steps'],r['reference_sha256'])
        if r['repeat']<1:
            rejected.append(dict(receipt=path.name,reason='Timing-harness preflight, not a final benchmark repeat'));continue
        if key not in allowed:
            rejected.append(dict(receipt=path.name,reason='Reference adapter not in final frozen selection'));continue
        if any(r['timing_started_epoch']-15<=t<=r['timing_finished_epoch']+15 for t in foreign):
            rejected.append(dict(receipt=path.name,reason='Foreign-context observation overlaps benchmark interval plus15s'));continue
        entry=allowed[key];r['cohort']=entry['cohort'];r['receipt']=path.name
        config_path=Path(entry['adapter']).parent/'config.json'
        cfg=json.loads(config_path.read_text());psha=cfg.get('predictor_sha256')
        r['inherited_preparation_seconds']=costs.get(psha) if psha else 0.
        groups[r['task'],entry['cohort'],r['recipe'],r['steps'],r['module']].append(r)
    assert groups,'No clean timing replays match frozen adapters'
    rows=[]
    for (task,cohort,recipe,steps,module),rr in sorted(groups.items()):
        expected={(1,101),(2,102),(3,103)}
        actual=[(r['repeat'],r['seed']) for r in rr]
        row=dict(task=task,cohort=cohort,recipe=recipe,module=module,steps=steps,repeats=len(rr),update_seeds=sorted({r['seed'] for r in rr}),all_weights_bitwise_equal=all(r['weights_bitwise_equal'] for r in rr),maximum_relative_weight_difference=max(r['relative_weight_difference'] for r in rr),receipts=[r['receipt'] for r in rr])
        row['registered_three_seed_schedule_complete']=len(actual)==3 and set(actual)==expected
        for field in ['online_seconds','prefix_seconds','standalone_pipeline_seconds','standalone_script_seconds']:
            row[field]=stats([r[field] for r in rr])
        row['online_plus_preparation_seconds']=stats([r['online_seconds']+r['prefix_seconds']+r.get('offline_initialization_seconds',0.) for r in rr])
        row['inherited_preparation_seconds']=[r['inherited_preparation_seconds'] for r in rr]
        row['cold_standalone_seconds']=stats([r['standalone_script_seconds']+r['inherited_preparation_seconds'] for r in rr]) if all(r['inherited_preparation_seconds'] is not None for r in rr) else None
        row['cold_total_scope']='Sum of standalone replay and recorded predictor-preparation components, not a separately timed end-to-end cold pipeline; preparation startup/data loading outside historical timers may be omitted.'
        row['eligible_for_direct_frozen_quality_speed_claim']=row['registered_three_seed_schedule_complete'] and row['all_weights_bitwise_equal']
        rows.append(row)
    write(EXP/'results/frozen_timing_analysis.json',dict(freeze_sha256=sha(EXP/'configs/final_freeze.json'),rows=rows,rejected_receipts=rejected,scope='Repeated standalone replays without intermediate research diagnostics. GPU occupancy observations exclude intervals with foreign contexts; samples are not continuous proof of exclusivity. Cold totals add recorded inherited predictor preparation, which may come from the historical implementation. All times exclude shared warmup training and downstream evaluation. A timing/quality claim additionally requires the paired frozen accuracy results. Non-identical replay weights remain explicit and cannot be silently assigned reference accuracy.'))
    log(dict(clean_timing_cohorts=len(rows),rejected_receipts=len(rejected)))


if __name__=='__main__':main()
