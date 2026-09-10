"""Collect exact parser-warning cases for human-readable mathematical review."""
from common import *
fp=S/'results/confirmation_plan.json';plan=json.loads(fp.read_text());entries=[];audits=[]
tags=[('confirmation_'+d,d) for d in plan['datasets']]+[('final_native_dev500','dev'),('final_primary_dev500','dev'),('final_selection_fixed16','dev'),('minerva_numeric_corrected','ood_minerva')]
for tag,dataset in tags:
 ap=S/f'audits/scoring_{tag}.json';a=json.loads(ap.read_text());assert a['script_sha256']==plan['minerva_numeric_scoring_script_sha256' if tag=='minerva_numeric_corrected' else 'strict_scoring_script_sha256']
 assert a['dataset_sha256']==sha(S/f'data/{dataset}.jsonl')
 data=read(S/f'data/{dataset}.jsonl');cache={};rows=[]
 for j,w in enumerate(a['parser_comparison_warnings']):
  index=w['index'];entry=dict(case_id=f'{tag}:{j}',tag=tag,dataset=dataset,warning=w,problem=data[index]['problem'],solution=data[index]['solution'])
  if 'model' in w:
   n=w['model'];path=Path(a['prediction_root'])/n/'math_predictions.jsonl'
   if n not in cache:cache[n]=read(path)
   assert a['models'][n]['predictions_sha256']==sha(path)
   r=cache[n][index];entry.update(model=n,prediction=r['prediction'],original_correct=bool(r['correct']),strict_correct=bool(a['models'][n]['strict_vector'][index]),finish_reason=r['finish_reason'],generated_tokens=r['generated_tokens'],predictions_sha256=sha(path))
  rows.append(entry);entries.append(entry)
 audits.append(dict(tag=tag,dataset=dataset,audit_sha256=sha(ap),n_models=len(a['models']),n_judgments=sum(m['n'] for m in a['models'].values()),warning_count=len(rows)))
jsonl(S/'audits/final_scoring_warning_cases.jsonl',entries)
write(S/'audits/final_scoring_warning_inventory.json',dict(time=time.time(),script_sha256=sha(__file__),plan_sha256=sha(fp),audits=audits,n_warning_events=len(entries),case_file_sha256=sha(S/'audits/final_scoring_warning_cases.jsonl'),scope='Preparation only. Every warning requires review or exact-link reuse of a prior mathematical review; this does not change original scores or mark allreviewed.'))
print(json.dumps(dict(n_warning_events=len(entries),audits=audits)))
