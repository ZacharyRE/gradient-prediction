import random,unicodedata,requests
import pyarrow.parquet as pq
from collections import Counter
from common import *
from math_verify import parse
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split()).lower()
train=read(S/'data/train.jsonl');historical=[]
for p in (ROOT/'data/MATH').glob('*/*.parquet'):historical+=pq.read_table(p).to_pylist()
seen={norm(r['problem']) for r in train+historical+read(S/'data/dev.jsonl')};audit={};allq=set(seen)
reviewp=S/'audits/ood_overlap_review.json'
removed={r['problem'] for r in json.loads(reviewp.read_text())['excluded']} if reviewp.exists() else set()
for name in ['minerva','olympiad','svamp','amc23']:
 rows=read(S/f'data/downloads/{name}_rows.jsonl');out=[];excluded=[]
 for i,r in enumerate(rows):
  if name=='svamp':q=r['question_concat'];answer=str(r['Answer'])
  elif name=='olympiad':
   if r['modality']!='Text-only' or r['language']!='English' or r['subject']!='Math' or r['is_multiple_answer'] or len(r['final_answer'])!=1:
    excluded.append(dict(index=i,reason='not_single_answer_english_text_math'));continue
   q=r['question'];answer=r['final_answer'][0].strip('$')
  else:q=r['question'];answer=str(r['answer'])
  if q in removed:excluded.append(dict(index=i,reason='manually_confirmed_duplicate'));continue
  if norm(q) in allq:excluded.append(dict(index=i,reason='exact_normalized_overlap'));continue
  solution='\\boxed{'+answer+'}'
  if not parse(solution):excluded.append(dict(index=i,reason='gold_unparseable'));continue
  allq.add(norm(q));out.append(dict(problem=q,solution=solution,answer=answer,source=name,source_index=i))
 jsonl(S/f'data/ood_{name}.jsonl',out);audit[name]=dict(n=len(out),excluded=excluded,sha256=sha(S/f'data/ood_{name}.jsonl'))
write(S/'audits/ood_reserved.json',dict(time=time.time(),datasets=audit,status='Reserved; no model evaluated; no recipe selection on these outcomes',scope='Mathematical task transfer, not universal nonmathematical capability',aggregation='Macro average of minerva, olympiad, svamp. AMC23 small-n supplementary only.'))
print(json.dumps(audit,indent=2))
