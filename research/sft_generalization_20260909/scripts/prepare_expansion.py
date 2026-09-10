"""Prepare, but do not generate/train, additional official MATH train questions."""
import random,re
import pyarrow.parquet as pq
from common import *
def norm(s):return re.sub(r'[^a-z0-9]','',s.lower())
def question(r):return r.get('problem',r.get('question',''))
def main():
 excluded=set();sources=[]
 def add(p,why):
  rr=read(p);qq={norm(question(r)) for r in rr if question(r)};excluded.update(qq)
  sources.append(dict(path=str(p),why=why,identified_questions=len(qq),sha256=sha(p)))
 add(S/'data/train.jsonl','Already used original training questions; generate no duplicate targets')
 add(S/'data/dev.jsonl','Current reused development questions')
 for entry in json.loads((FOLLOW/'audits/holdout_provenance.json').read_text())['sources']:
  p=ROOT/entry['path'];name=p.name.lower()
  if any(x in name for x in ['dev','test','target','unseen']) or entry['path'].startswith('predictor/'):
   add(p,'Historical identified dev/test/target material or predictor split, conservatively excluded')
 for p in (S/'data').glob('ood_*.jsonl'):add(p,'Reserved OOD confirmation')
 test=pq.read_table(ROOT/'data/MATH/test/test-00000-of-00001.parquet').to_pylist();excluded.update(norm(r['problem']) for r in test)
 raw=pq.read_table(ROOT/'data/MATH/train/train-00000-of-00001.parquet').to_pylist();rows=[];seen=set();filtered=0
 for i,r in enumerate(raw):
  key=norm(r['problem'])
  if key in excluded or key in seen:filtered+=1;continue
  seen.add(key);rows.append(dict(r,source='DigitalLearningGmbH/MATH-lighteval:train',source_row_index=i,role='additional_training_only'))
 random.Random(20260915).shuffle(rows);jsonl(S/'data/expansion_pool.jsonl',rows)
 write(S/'audits/expansion_pool.json',dict(time=time.time(),official_train=7500,retained=len(rows),filtered=filtered,seed=20260915,sha256=sha(S/'data/expansion_pool.jsonl'),exclusions=sources,all_official_test_excluded=True,status='Prepared only, no target generation or training yet',limitation='Alphanumeric exact normalization and identifiable historical sources; not a pretraining or exhaustive semantic contamination certificate'))
 print('Prepared additional training pool:',len(rows))
if __name__=='__main__':main()
