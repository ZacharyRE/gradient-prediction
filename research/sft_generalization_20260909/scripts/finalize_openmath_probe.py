"""Decontamination and nested size datasets; still requires target process review."""
import random
from collections import Counter,defaultdict
import pyarrow.parquet as pq,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import *
rows=read(S/'data/openmath_external_candidates.jsonl');refs=[]
for split in ['train','test']:refs+=pq.read_table(ROOT/f'data/MATH/{split}/{split}-00000-of-00001.parquet').to_pylist()
refs+=read(S/'data/dev.jsonl')
for p in (S/'data').glob('ood_*.jsonl'):refs+=read(p)
refs=list({r['problem']:r for r in refs}.values());v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,dtype=np.float32);x=v.fit_transform([r['problem'] for r in rows+refs]);rejected=[];kept=[]
for start in range(0,len(rows),128):
 sim=(x[start:min(start+128,len(rows))]@x[len(rows):].T).toarray();ix=sim.argmax(1)
 for j,k in enumerate(ix):
  score=float(sim[j,k]);i=start+j
  if score>=.8:rejected.append(dict(index=i,reference_index=int(k),score=score,problem=rows[i]['problem'],reference_problem=refs[k]['problem']))
  else:kept.append(dict(rows[i],max_reference_similarity=score))
random.Random(20260930).shuffle(kept);decision=S/'audits/openmath_target_review_decisions.json';excluded=set(json.loads(decision.read_text())['excluded_problem_hashes']) if decision.exists() else set();kept=[r for r in kept if hashlib.sha256(r['problem'].encode()).hexdigest() not in excluded];chosen=kept[:12000];assert len(chosen)>=8000
sets={'openmath_external_large':chosen,'openmath_external_small':chosen[:2000]};audit={}
for name,rr in sets.items():
 jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(r['target_tokens'] for r in rr),sources=dict(Counter(r['problem_source'] for r in rr)))
review=S/'audits/openmath_target_review.jsonl';groups=defaultdict(list);med=float(np.median([r['target_tokens'] for r in chosen if r['problem_source']=='augmented_math']))
for r in chosen:
 group=r['problem_source']
 if group=='augmented_math':group+='_long' if r['target_tokens']>med else '_short'
 groups[group].append(r)
if not review.exists():
 rng=random.Random(20261001);samples=[]
 for g,rr in sorted(groups.items()):
  for r in rng.sample(rr,min(8,len(rr))):samples.append(dict(r,review_group=g,problem_sha256=hashlib.sha256(r['problem'].encode()).hexdigest()))
 jsonl(review,samples)
write(S/'audits/openmath_similarity_rejections.json',rejected)
write(S/'audits/openmath_target_build.json',dict(time=time.time(),initial=16000,similarity_rejected=len(rejected),retained_before_size_limit=len(kept),review_excluded=sorted(excluded),reference_n=len(refs),reference_content_sha256=hashlib.sha256(json.dumps([r['problem'] for r in refs],ensure_ascii=False).encode()).hexdigest(),threshold=.8,datasets=audit,review_group_sizes={g:len(rr) for g,rr in groups.items()},status='Provisional, review and release required',limitations='Similarity excludes all original MATH12500 as well as reservedOOD, conservatively removing some legitimate related problems. Synthetic majority expected_answer is not human gold. Missing-box/unparsed filters are not estimates of erroneous answer prevalence. Small is nested first2000; large is first12000 after frozen shuffle, not full-corpus representative.'))
print(json.dumps(dict(n=len(chosen),similarity_rejections=len(rejected),datasets=audit,review_groups={g:len(rr) for g,rr in groups.items()}),indent=2))
