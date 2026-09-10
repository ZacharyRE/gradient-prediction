import re
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from common import *
def norm(q):return re.sub(r'[^a-z0-9]','',q.lower())
prior=[]
for p in (ROOT/'data/MATH').glob('*/*.parquet'):
 for r in pq.read_table(p).to_pylist():prior.append(dict(problem=r['problem'],source=str(p)))
rows=[]
for name in ['minerva','olympiad','svamp','amc23']:
 for i,r in enumerate(read(S/f'data/ood_{name}.jsonl')):rows.append(dict(r,dataset=name,index=i))
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=150000,dtype=np.float32)
x=v.fit_transform([r['problem'] for r in prior+rows]);p=x[:len(prior)];z=x[len(prior):];matches=[]
for i in range(len(rows)):
 scores=(z[i]@p.T).toarray()[0];j=int(scores.argmax());score=float(scores[j])
 if score>.8 or norm(rows[i]['problem'])==norm(prior[j]['problem']):matches.append(dict(dataset=rows[i]['dataset'],index=rows[i]['index'],similarity=score,problem=rows[i]['problem'],prior=prior[j]))
write(S/'audits/ood_near_overlap.json',dict(method='char3-5 TFIDF maximum cosine against all 12500 official MATH train/test questions; >0.8 candidates for manual review, not automatic semantic proof',n_prior=len(prior),n_ood=len(rows),matches=matches))
print(json.dumps(matches,ensure_ascii=False,indent=2))
