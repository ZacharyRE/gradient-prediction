"""Apply the prospectively registered quality/time gate to completed pilots."""
from common import *

def main():
    path=EXP/'provenance/late_confirmation_gate_registration.json'
    gate=json.loads(path.read_text())
    summary_path=EXP/'results/late_development_summary.json'
    summary=json.loads(summary_path.read_text())
    expected={(m,s) for m in gate['eligible_methods'] for s in gate['eligible_steps']}
    rows=[r for r in summary['groups'] if (r['method'],r['steps']) in expected]
    if {(r['method'],r['steps']) for r in rows}!=expected or any(sorted(r['starts'])!=[301,311,321] for r in rows):
        log(dict(decision_pending=True,reason='All registered three-start readouts must finish before a decision.'))
        return
    out=EXP/'results/late_confirmation_gate_decision.json'
    if out.exists():
        old=json.loads(out.read_text());assert old['candidate_rows']==rows and old['gate_sha256']==sha(path)
        log(old);return
    passing=[r for r in rows if r['mean_accuracy_pct']>=gate['minimum_development_accuracy_pct']
             and r['mean_preparation_updates_seconds']<=gate['maximum_development_preparation_updates_seconds']]
    passing.sort(key=lambda r:(-r['mean_accuracy_pct'],r['mean_preparation_updates_seconds'],r['method'],r['steps']))
    remaining=json.loads((EXP/'configs/run_deadline.json').read_text())['deadline_epoch']-time.time()
    selected=passing[0] if passing and remaining>=gate['minimum_seconds_before_user_deadline_at_decision'] else None
    result=dict(complete=True,epoch=time.time(),gate_sha256=sha(path),development_summary_sha256=sha(summary_path),
                candidate_rows=rows,passing_rows=passing,selected=selected,remaining_seconds=remaining,
                triggered=selected is not None,
                reason='Quality and time gate passed.' if selected else 'No point passed both quality and time thresholds.' if not passing else 'Insufficient remaining confirmation budget.',
                scope='Conditional confirmation decision for a previously registered method; original frozen results unchanged.')
    shutil.copy2(summary_path,EXP/'provenance/late_confirmation_development_snapshot.json')
    write(out,result);log(result)

if __name__=='__main__':main()
