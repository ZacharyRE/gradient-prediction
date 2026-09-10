"""Calibrate the quarantined external-data judge; create a review queue, never train."""
from collections import Counter,defaultdict
import random
from common import *

root=S/'results/process_judge/openmath_external'
assert (root/'complete.json').exists(),'External judge is not complete'
src=S/'data/openmath_external_large.jsonl';rows=read(src);pred=read(root/'predictions.jsonl')
plan=json.loads((S/'audits/openmath_external_judge_plan.json').read_text())
decpath=S/'audits/openmath_target_review_decisions.json';review=read(S/'audits/openmath_target_review.jsonl');dec=json.loads(decpath.read_text())
assert sha(src)==plan['input_sha256']==json.loads((root/'manifest.json').read_text())['input_sha256']
assert sha(decpath)==plan['review_decisions_sha256']
assert len(rows)==len(pred)==12000 and len(review)==len(dec['records'])==32
def digest_text(t):return hashlib.sha256(t.encode()).hexdigest()
def label(p):return 'incomplete' if p['finish_reason']!='stop' else (p['judgment'] or {}).get('verdict','unparsed')
lookup={};groups=defaultdict(Counter)
for i,(r,p) in enumerate(zip(rows,pred)):
 key=digest_text(r['problem']);assert p['index']==i and p['problem_sha256']==key and p['target_sha256']==digest_text(r['solution'])
 assert key not in lookup;lookup[key]=(r,p)
 groups[r['problem_source']][label(p)]+=1
cal=[];counts=Counter()
for d,r in zip(dec['records'],review):
 key=d['problem_sha256'];assert key==digest_text(r['problem'])
 actual,p=lookup[key];assert actual['solution']==r['solution']
 assistant='exclude' if d['decision'].startswith('exclude') else 'retain';judge=label(p);counts[assistant,judge]+=1
 cal.append(dict(d,assistant_group=assistant,judge_label=judge,judge=p['judgment'],target_sha256=digest_text(r['solution'])))
# Conditional audit of accepted weak labels, stratified rather than prevalence-weighted.
known={r['problem_sha256'] for r in dec['records']};pool=defaultdict(list)
extra=S/'audits/openmath_additional_known_issues.json'
if extra.exists():known.update(r['problem_sha256'] for r in json.loads(extra.read_text())['records'])
for r,p in zip(rows,pred):
 if p['problem_sha256'] in known or label(p)!='sound':continue
 group=r['problem_source']
 if group=='augmented_math':group+='_'+('long' if r['target_tokens']>=600 else 'short')
 pool[group].append(dict(r,problem_sha256=p['problem_sha256'],target_sha256=p['target_sha256'],judge_label='sound',judge=p['judgment'],review_group=group))
rng=random.Random(20261018);todo=[]
for group,rr in sorted(pool.items()):todo.extend(rng.sample(rr,min(6,len(rr))))
destination=S/'audits/openmath_judge_sound_review.jsonl'
if destination.exists():assert read(destination)==todo
else:jsonl(destination,todo)
out=dict(time=time.time(),input_sha256=sha(src),judge_predictions_sha256=sha(root/'predictions.jsonl'),calibration=cal,calibration_counts=[dict(assistant=a,judge=b,n=n) for (a,b),n in counts.items()],source_weak_counts={k:dict(v) for k,v in groups.items()},followup_review=dict(path=str(destination),n=len(todo),sha256=sha(destination),strata={g:len(v) for g,v in pool.items()}),status='No data or training release. Next review is conditioned on judge-sound labels and stratified, not a representative estimate of the full12k error rate.',limitation='Assistant reviews are not independent human gold; inspect disagreement severity and exact counterexamples. Public-data quality is not itself evidence for the cause of previous local SFT failures.')
write(S/'audits/openmath_judge_calibration.json',out)
print(json.dumps(dict(counts=out['calibration_counts'],source_counts=out['source_weak_counts'],next_review_n=len(todo),sound_among_excluded=[r for r in cal if r['assistant_group']=='exclude' and r['judge_label']=='sound']),indent=2))
