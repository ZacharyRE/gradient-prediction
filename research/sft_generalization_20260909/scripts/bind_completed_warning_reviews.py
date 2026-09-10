"""Bind already-reviewed exact cases; refuse to approve any unreviewed warning."""
import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--scope',choices=['final','supplement'],required=True)
p.add_argument('--review-pair',action='append',nargs=2,metavar=('PREVIEW_CASES','REVIEW'),required=True)
a=p.parse_args();prefix='final' if a.scope=='final' else 'supplement'
inventory_path=S/f'audits/{prefix}_scoring_warning_inventory.json'
cases_path=S/f'audits/{prefix}_scoring_warning_cases.jsonl'
inventory=json.loads(inventory_path.read_text());cases=read(cases_path)
assert inventory['case_file_sha256']==sha(cases_path)
assert inventory['n_warning_events']==len(cases)
known={};sources={str(inventory_path.relative_to(S)):sha(inventory_path),str(cases_path.relative_to(S)):sha(cases_path)}
for cp,rp in a.review_pair:
 cp=S/cp;rp=S/rp;preview=read(cp);review=json.loads(rp.read_text())
 assert review['preview_cases_sha256']==sha(cp)
 sources[str(cp.relative_to(S))]=sha(cp);sources[str(rp.relative_to(S))]=sha(rp)
 for c,d in zip(preview,review['records']):
  assert c['case_id']==d['case_id'] and d['reviewed'] and d['complete_text_read']
  assert d['case_sha256']==hashlib.sha256(json.dumps(c,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
  assert d['prediction_sha256']==hashlib.sha256(c['prediction'].encode()).hexdigest()
  assert c['case_id'] not in known
  known[c['case_id']]=(c,d,str(rp.relative_to(S)))
 assert len(preview)==len(review['records'])
records=[];missing=[]
for c in cases:
 if c['case_id'] not in known:
  missing.append(c['case_id']);continue
 previous,d,path=known[c['case_id']]
 for key in ['case_id','tag','warning','model','problem','solution','prediction','predictions_sha256','original_correct','strict_correct']:
  assert c[key]==previous[key],(c['case_id'],key)
 assert c['warning']['index']==previous['index']
 records.append({'case_id':c['case_id'],'reviewed':True,'review_source':path,'review_source_sha256':sha(S/path),'case_sha256':hashlib.sha256(json.dumps(c,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'decision':d})
assert not missing, f'Unreviewed warnings remain: {missing}'
for audit in inventory['audits']:
 path=S/f"audits/scoring_{audit['tag']}.json"
 assert sha(path)==audit['audit_sha256'];sources[str(path.relative_to(S))]=sha(path)
out={'time':time.time(),'all_reviewed':True,'plan_sha256':inventory['plan_sha256'],'inventory_sha256':sha(inventory_path),'case_file_sha256':sha(cases_path),'n_warning_events':len(cases),'records':records,'source_sha256':sources,'script_sha256':sha(__file__),'scope':'Exact full-text and prediction-file identity binding of prior assistant mathematical reviews; not an independent human review. No unreviewed warning automatically approved. Zero-warning audits included with hashes. Original scores unchanged; any reference-quality concern retained in original case decision.'}
destination=S/f'audits/{prefix}_scoring_warning_review.json';assert not destination.exists()
write(destination,out);print(json.dumps({'reviewed':len(records),'destination':str(destination)}))
