"""Keep calibration annotations outside the judge's prompt/input data."""
from common import *
rows=[];seen=set();calibration=[]
for stem in ['guided','target']:
 review=read(S/f'audits/{stem}_review.jsonl')
 decisions=json.loads((S/f'audits/{stem}_review_decisions.json').read_text())['decisions']
 assert len(review)==len(decisions)
 for r,d in zip(review,decisions):
  assert r['original_index']==d['original_index']
  key=(r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest())
  calibration.append(dict(original_index=key[0],target_sha256=key[1],review_set=stem,decision=d['decision'],reason=d['reason']))
  if key not in seen:
   seen.add(key);rows.append({k:r[k] for k in ['original_index','target_source','problem','solution']})
for r in read(S/'data/guided_all.jsonl'):
 key=(r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest())
 if key not in seen:
  seen.add(key);rows.append({k:r[k] for k in ['original_index','target_source','problem','solution']})
jsonl(S/'data/judge_guided_input.jsonl',rows)
write(S/'audits/judge_guided_calibration.json',dict(n=len(rows),input_sha256=sha(S/'data/judge_guided_input.jsonl'),calibration=calibration,reviewer='AI research assistant; specific counterexamples separately checked by exact computation; not independent human gold labels',time=time.time()))
print(json.dumps(dict(n=len(rows),calibration=len(calibration))))
