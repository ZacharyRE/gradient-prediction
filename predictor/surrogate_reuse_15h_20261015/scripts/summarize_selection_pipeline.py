"""Qualify complete fresh selection workflows; timing repeats are not new seeds."""
from common import *
from collections import defaultdict
import csv,sys

def main():
    policy=EXP/'configs/surrogate_cost_policy.json';plan=json.loads(policy.read_text())['measured_selection_pipeline_plan'];groups=defaultdict(list)
    for p in (EXP/'results/selection_pipeline').glob('*.json'):
        r=json.loads(p.read_text());assert r['complete'] and r['fresh_predictions_match'] and r['selected_candidate_matches'] and r['cost_policy_sha256']==sha(policy)
        groups[r['family'],r['condition']].append((p,r))
    rows=[];complete=True
    for family in plan['families']:
        for condition in plan['conditions']:
            repeats={}
            for p,r in sorted(groups[family,condition['name']],key=lambda x:x[0].name):repeats[r['repeat']]=(p,r)
            passed=sorted(repeats)==list(range(1,plan['repetitions']+1));complete=complete and passed;times=[r['pipeline_script_seconds'] for _,r in repeats.values()]
            rows.append(dict(family=family,condition=condition['name'],method=condition['method'],steps=condition['steps'],selector=condition['selector'],warmup_seed=plan['warmup_seed'],qualified_repetitions=len(repeats),three_repeats_complete=passed,pipeline_seconds_median=float(np.median(times)) if times else None,pipeline_seconds_min=min(times) if times else None,pipeline_seconds_max=max(times) if times else None,selected_subsets=sorted({r['selected_subset'] for _,r in repeats.values()}),receipts=[dict(path=str(p),sha256=sha(p)) for p,_ in repeats.values()]))
    result=dict(complete=complete,cost_policy_sha256=sha(policy),rows=rows,scope=plan['scope'],qualification=plan['qualification'],important='One scientific warmup/update-seed pair and one candidate family, with repeated fresh workflow timing. This does not estimate across-start timing or generalize workflow cost to genre subsets. Once-only predictor preparation and the common target warmup are additional. Predictions and selected candidates reproduce the pre-existing frozen choices; no recipe is selected from these timings.')
    write(EXP/'results/selection_pipeline_summary.json',result)
    if rows:
        flat=[{k:v for k,v in r.items() if k not in ['receipts','selected_subsets']} for r in rows]
        with (EXP/'results/selection_pipeline_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    log(dict(complete=complete,conditions=len(rows),qualified=sum(r['three_repeats_complete'] for r in rows)))
    if complete and (EXP/'results/final_selection_analysis.json').exists():
        for script in ['surrogate_cost_comparison.py','render_surrogate_results.py']:
            subprocess.run([sys.executable,str(CODE_ROOT/script)],check=True,cwd=ROOT)

if __name__=='__main__':main()
