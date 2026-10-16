"""Join frozen test accuracy, independently timed replays, and matrix-FLOP receipts."""
from common import *
import csv


def main():
    freeze=json.loads((EXP/'configs/final_freeze.json').read_text())
    accuracy=json.loads((EXP/'results/frozen_test_analysis.json').read_text())
    timing=json.loads((EXP/'results/frozen_timing_analysis.json').read_text())
    assert accuracy['freeze_sha256']==timing['freeze_sha256']==sha(EXP/'configs/final_freeze.json')
    comparisons={(r['task'],r['cohort']):r for r in accuracy['comparisons']}
    profiles=[json.loads(p.read_text()) for p in (EXP/'results/flops').glob('*.json')]
    rows=[]
    for result in accuracy['results']:
        task=result['task'];cohort=result['cohort'];entries=[e for e in freeze['entries'] if e['task']==task and e['cohort']==cohort]
        matches=[r for r in timing['rows'] if r['task']==task and r['cohort']==cohort]
        assert len(matches)<=1,(task,cohort,'Choose one execution module in the frozen timing plan')
        timed=matches[0] if matches else None;comparison=comparisons.get((task,cohort))
        reference=freeze['comparison_reference'][task]
        gap=0. if cohort==reference else comparison.get('mean_gap_pp') if comparison else None
        interval=[0.,0.] if cohort==reference else comparison.get('ci95_gap_pp',[None,None]) if comparison else [None,None]
        row=dict(task=task,cohort=cohort,primary_candidate=result['primary_candidate'],update_steps=0 if cohort=='warmup' else entries[0]['steps'],lr=entries[0]['lr'],update_seeds=','.join(map(str,result['seeds'])),accuracy_pct=result['mean_pct'],accuracy_ci95_low_pct=result['ci95_pct'][0],accuracy_ci95_high_pct=result['ci95_pct'][1],reference=reference,gap_pp=gap,gap_ci95_low_pp=interval[0],gap_ci95_high_pp=interval[1],online_seconds_median=None,online_plus_preparation_seconds_median=None,standalone_script_seconds_median=None,standalone_script_seconds_min=None,standalone_script_seconds_max=None,inherited_predictor_preparation_seconds_mean=None,timed_replays=0,timing_weights_match=False,trainer_module=None,total_matrix_pflops=None,online_matrix_pflops=None)
        if timed:
            row.update(online_seconds_median=timed['online_seconds']['median'],online_plus_preparation_seconds_median=timed['online_plus_preparation_seconds']['median'],standalone_script_seconds_median=timed['standalone_script_seconds']['median'],standalone_script_seconds_min=timed['standalone_script_seconds']['minimum'],standalone_script_seconds_max=timed['standalone_script_seconds']['maximum'],timed_replays=timed['repeats'],timing_weights_match=timed['all_weights_bitwise_equal'],trainer_module=timed['module'])
            prep=timed['inherited_preparation_seconds'];row['inherited_predictor_preparation_seconds_mean']=float(np.mean(prep)) if all(v is not None for v in prep) else None
            ff=[f for f in profiles if f['task']==task and f['recipe']==timed['recipe'] and f['module']==timed['module'] and f['steps']==timed['steps'] and f['seed']==101]
            if ff:
                assert len({f['total_matrix_flops'] for f in ff})==1,'Investigate inconsistent FLOP replays'
                row['total_matrix_pflops']=ff[0]['total_matrix_flops']/1e15
                if ff[0]['online_matrix_flops'] is not None:row['online_matrix_pflops']=ff[0]['online_matrix_flops']/1e15
        row['eligible_for_direct_frozen_quality_speed_claim']=bool(timed and timed['eligible_for_direct_frozen_quality_speed_claim'])
        rows.append(row)
    with (EXP/'results/final_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write(EXP/'results/final_comparison.json',dict(freeze_sha256=accuracy['freeze_sha256'],rows=rows,scope='Frozen test accuracy and paired group/seed bootstrap intervals. Times are medians of clean replays with explicitly recorded weight matches. Predictor preparation is an additional historical component, not included in online/standalone times. FLOPs cover measured matrix operations in seed101 adaptation and cache preparation, excluding inherited offline fitting and nonlinear/scalar/optimizer operations. Empty cells are unmeasured, not zero. These tables do not alone establish superiority over independently tuned cheap controls.'))
    log(dict(final_rows=len(rows)))


if __name__=='__main__':main()
