"""Calibrate the selected-target judge on unique previously reviewed traces."""
from collections import Counter
from common import *
root=S/'results/process_judge/selected_sources';assert (root/'complete.json').exists()
source=S/'data/judge_selected_sources_input.jsonl';inputs=read(source);predictions=read(root/'predictions.jsonl')
assert json.loads((root/'manifest.json').read_text())['input_sha256']==sha(source)
assert len(inputs)==len(predictions)==4766
pred={}
for i,(a,b) in enumerate(zip(inputs,predictions)):
 assert b['index']==i and a['original_index']==b['original_index'] and a['target_sha256']==b['target_sha256']
 pred[(b['original_index'],b['target_sha256'])]=b
def label(r):return 'incomplete' if r['finish_reason']!='stop' else (r['judgment'] or {}).get('verdict','unparsed')
decisions=json.loads((S/'audits/nll_noguided_review_decisions.json').read_text());truth={};raw_records=decisions['records']
for r in raw_records:
 key=(r['original_index'],r['target_sha256']);decision='exclude' if r['decision'].startswith('exclude') else 'retain'
 if key in truth:assert truth[key]['assistant']==decision
 else:truth[key]=dict(original_index=key[0],target_sha256=key[1],assistant=decision,assistant_reason=r['reason'])
counts=Counter();calibration=[]
for key,r in truth.items():
 p=pred[key];v=label(p);counts[r['assistant'],v]+=1;calibration.append(dict(r,judge_label=v,judge=p['judgment'],finish_reason=p['finish_reason']))
sources={}
for dataset in ['sample_all','teacher_all','teacher32_all']:
 rr=read(S/f'data/{dataset}.jsonl');cc=Counter();by_historical=Counter()
 for r in rr:cc[label(pred[(r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest())])]+=1
 sources[dataset]=dict(n=len(rr),weak_verdicts=dict(cc),dataset_sha256=sha(S/f'data/{dataset}.jsonl'))
out=dict(time=time.time(),raw_calibration_records=len(raw_records),unique_calibration_targets=len(truth),calibration_counts=[dict(assistant=a,judge=b,n=n) for (a,b),n in counts.items()],calibration=calibration,sources=sources,judge_predictions_sha256=sha(root/'predictions.jsonl'),status='Calibration and weak-label audit only; no newly filtered training data released.',limitations=['Assistant process review is not independent-human gold. Calibration was selected by earlier NLL-target reviews, not representative of all source traces.','Source sound/unsound percentages are judge outputs, not measured true error prevalence. Correct final answers and judge-sound decisions do not certify derivations.','Repeated target keys are counted once in calibration to avoid pseudo-replication.'])
write(S/'audits/selected_judge_calibration.json',out)
print(json.dumps(dict(unique_calibration=len(truth),counts=out['calibration_counts'],sources=sources,missed_invalid=[r for r in calibration if r['assistant']=='exclude' and r['judge_label']=='sound']),indent=2))
