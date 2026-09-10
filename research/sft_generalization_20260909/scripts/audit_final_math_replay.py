"""Cross-check overlapping reused1500 outputs inside frozen full5000 evaluation."""
from common import *
planpath=S/'results/confirmation_plan.json';plan=json.loads(planpath.read_text())
assert sha(__file__)==plan['math_replay_script_sha256']
models=json.loads((S/'results/confirmation_manifest.json').read_text());root=S/'results/evaluation'
missing=[n for n in models if not (root/'math_reused5000'/n/'math_summary.json').exists()]
if missing:print(json.dumps(dict(pending=missing)));raise SystemExit(0)
full=read(S/'data/math_reused5000.jsonl');subset=read(S/'data/math_reused1500.jsonl')
indices=[r['source_test_index'] for r in subset];assert len(set(indices))==1500
assert all((full[i]['problem'],full[i]['solution'])==(r['problem'],r['solution']) for i,r in zip(indices,subset))
fields=['sample_hash','prediction','correct','finish_reason','generated_tokens'];rows=[]
for name in models:
 old=root/'math_reused1500'/name
 if not (old/'math_summary.json').exists():continue
 new=root/'math_reused5000'/name;rr=read(new/'math_predictions.jsonl');ss=read(old/'math_predictions.jsonl')
 assert len(rr)==5000 and len(ss)==1500
 nm=json.loads((new/'math_manifest.json').read_text());om=json.loads((old/'math_manifest.json').read_text())
 for k in ['model','adapter','engine','batch_invariant','max_tokens','temperature','chunk_size','script_sha256']:assert nm[k]==om[k],(name,k)
 counts={k:sum(rr[i][k]==s[k] for i,s in zip(indices,ss)) for k in fields}
 mismatch=[dict(full_index=i,subset_index=j,fields=[k for k in fields if rr[i][k]!=s[k]]) for j,(i,s) in enumerate(zip(indices,ss)) if any(rr[i][k]!=s[k] for k in fields)]
 rows.append(dict(model=name,n=1500,equal_counts=counts,first_mismatches=mismatch[:10],mismatch_count=len(mismatch),full_predictions_sha256=sha(new/'math_predictions.jsonl'),subset_predictions_sha256=sha(old/'math_predictions.jsonl')))
assert any(r['model']=='base' for r in rows)
passed=all(r['mismatch_count']==0 for r in rows)
write(S/'audits/final_math_overlap_replay.json',dict(time=time.time(),passed=passed,script_sha256=sha(__file__),plan_sha256=sha(planpath),models=rows,unavailable_prior_models=[n for n in models if not (root/'math_reused1500'/n/'math_summary.json').exists()],scope='Specific overlapping question/full-text replay under fixed batch-invariant protocol. Both MATH datasets historically reused. Does not create independent confirmation.'))
assert passed,'Overlapping fullMATH/subset inference differs; inspect before interpretation'
print(json.dumps(dict(passed=passed,models=len(rows),question_model_pairs=1500*len(rows))))
