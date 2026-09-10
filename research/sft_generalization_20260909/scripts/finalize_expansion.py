"""Conservative training-only expansion filtering before any generation."""
import pyarrow.parquet as pq,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from common import *
from training.train_lora_sft import chat_prompt_ids
rows=read(S/'data/expansion_pool.jsonl');held=pq.read_table(ROOT/'data/MATH/test/test-00000-of-00001.parquet').to_pylist()+read(S/'data/dev.jsonl')
for p in (S/'data').glob('ood_*.jsonl'):held+=read(p)
held=list({r['problem']:r for r in held}.values());texts=[r['problem'] for r in rows+held]
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,dtype=np.float32);x=v.fit_transform(texts)
similar=[];maximum=[]
for start in range(0,len(rows),128):
 sim=(x[start:min(start+128,len(rows))]@x[len(rows):].T).toarray();idx=sim.argmax(1)
 for j,k in enumerate(idx):
  score=float(sim[j,k]);maximum.append(score)
  if score>=.8:similar.append(dict(index=start+j,heldout_index=int(k),similarity=score,training_problem=rows[start+j]['problem'],heldout_problem=held[k]['problem']))
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);teacher=AutoTokenizer.from_pretrained('/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137',local_files_only=True)
from evaluation.evaluate_sft_math_benchmarks import SYSTEM_PROMPT,USER_TEMPLATE
retained=[];reasons=[];lengths=[]
for i,r in enumerate(rows):
 student_length=len(chat_prompt_ids(tok,r['problem']))+len(tok.encode(r['solution'],add_special_tokens=False))+1
 prompt=teacher.apply_chat_template([dict(role='system',content=SYSTEM_PROMPT),dict(role='user',content=USER_TEMPLATE.format(problem=r['problem']))],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 teacher_length=len(teacher.encode(prompt,add_special_tokens=False))
 reason=[]
 if maximum[i]>=.8:reason.append('conservative_similarity_screen')
 if student_length>2048:reason.append('paired_raw_student_length')
 if teacher_length+2048>4096:reason.append('teacher_context_length')
 if reason:reasons.append(dict(index=i,reasons=reason));continue
 retained.append(r);lengths.append(dict(student=student_length,teacher=teacher_length))
jsonl(S/'data/expansion_ready.jsonl',retained)
write(S/'audits/expansion_similarity_candidates.json',similar)
write(S/'audits/expansion_release_for_generation.json',dict(time=time.time(),n=len(retained),input_n=len(rows),input_sha256=sha(S/'data/expansion_pool.jsonl'),output_sha256=sha(S/'data/expansion_ready.jsonl'),similarity_threshold=.8,heldout_n=len(held),heldout_content_sha256=hashlib.sha256(json.dumps([r['problem'] for r in held],ensure_ascii=False).encode()).hexdigest(),excluded=reasons,max_teacher_prompt=max(r['teacher'] for r in lengths),max_paired_student_length=max(r['student'] for r in lengths),status='Generation only, training requires target filtering/review',limitation='All candidates above character-TFIDF threshold excluded conservatively, including potentially distinct similar problems. This is not exhaustive semantic or pretraining decontamination.'))
print('Expansion ready',len(retained),'of',len(rows),'similarity candidates',len(similar))
