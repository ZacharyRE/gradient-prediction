"""Reuse an exact previously reviewed timeout case; nine-contrast Holm sensitivity."""
from scipy.stats import binomtest
from common import *
ap=S/'audits/coverage_math1500_analysis.json';analysis=json.loads(ap.read_text())
sp=S/'audits/scoring_coverage_math1500.json';scoring=json.loads(sp.read_text())
assert analysis['scoring_audit_sha256']==sha(sp)
reviewpath=S/'audits/strict_math1500_timeout_review.json';review=json.loads(reviewpath.read_text())
warnings=[]
for warning in scoring['parser_comparison_warnings']:
    assert warning['model']=='teacher_all_lr5e5_s43_epoch4' and warning['index']==636
    assert warning['message']=='Timeout during comparison' and warning['final_strict_correct'] is False
    path=S/'results/evaluation/math_reused1500'/warning['model']/'math_predictions.jsonl'
    assert sha(path)==scoring['models'][warning['model']]['predictions_sha256']
    row=read(path)[636];old=review['case']['prediction']
    for key in ['sample_hash','prediction','correct','finish_reason','generated_tokens']:assert row[key]==old[key]
    assert row['correct'] is False and scoring['models'][warning['model']]['strict_vector'][636]==0
    warnings.append(dict(warning=warning,exact_prior_review_match=True,
                         prior_review_sha256=sha(reviewpath),prior_reason=review['case']['reason'],
                         action='No score change; the same previously checked enormous exponent is not224.'))
results={}
for label,group in analysis['results'].items():
    rows=[]
    for pair in group['pairs']:
        w,l=pair['wins'],pair['losses']
        rows.append(dict(treatment=pair['treatment'],control=pair['control'],delta_pp=pair['delta_pp'],
                         wins=w,losses=l,p=binomtest(w,w+l).pvalue if w+l else 1.))
    assert len(rows)==9
    previous=0.
    for index,row in enumerate(sorted(rows,key=lambda x:x['p'])):
        previous=max(previous,min(1.,row['p']*(len(rows)-index)));row['holm9_p']=previous
    results[label]=rows
write(S/'audits/coverage_math1500_sensitivity.json',dict(time=time.time(),script_sha256=sha(__file__),
    analysis_sha256=sha(ap),scoring_audit_sha256=sha(sp),all_warnings_reviewed=True,
    warnings=warnings,paired_multiplicity=results,
    scope='Holm covers the nine fixed contrasts in this six-pilot follow-up, not all adaptive development '
          'selection. Single seed and historically reused questions. Existing unadjusted intervals and '
          'original scores remain unchanged.'))
print(json.dumps(dict(warnings=len(warnings),holm9_below05={k:[r for r in v if r['holm9_p']<.05] for k,v in results.items()}),indent=2))
