"""Bounded public-data scale probe, preparation only; no OOD answers used."""
import random,re
from collections import Counter
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from math_verify import parse,verify
from common import *
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents

def norm(s):return re.sub('[^a-z0-9]','',s.lower())
meta=json.loads((S/'audits/openmath_source_metadata.json').read_text());revision=meta['revision'];out=S/'data/openmath_external_candidates.jsonl';assert not out.exists()
# A single212.9MB parquet shard is the maximum remote file accessed, not the14M corpus.
url=f'https://huggingface.co/datasets/nvidia/OpenMathInstruct-2/resolve/{revision}/'+meta['files'][0]['name']
stream=load_dataset('parquet',data_files={'train':url},split='train',streaming=True).shuffle(seed=20260929,buffer_size=4096)
excluded=set()
for split in ['train','test']:
 for r in pq.read_table(ROOT/f'data/MATH/{split}/{split}-00000-of-00001.parquet').to_pylist():excluded.add(norm(r['problem']))
for p in (S/'data').glob('ood_*.jsonl'):
 for r in read(p):excluded.add(norm(r['problem']))
for entry in json.loads((FOLLOW/'audits/holdout_provenance.json').read_text())['sources']:
 p=ROOT/entry['path']
 if p.exists():
  for r in read(p):
   q=r.get('problem',r.get('question',''))
   if q:excluded.add(norm(q))
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);kept=[];seen=set();counts=Counter();scanned=0
for i,r in enumerate(stream):
 if i>=50000:break
 scanned+=1;key=norm(r['problem']);text=r['generated_solution']
 if key in excluded or key in seen:counts['duplicate_or_known_question']+=1;continue
 if any(x in text for x in ['[asy]','\\begin{asy}','```asy','<think>','</think>']):counts['code_or_thinking']+=1;continue
 boxes=boxed_contents(text)
 if not boxes or not verify(parse(str(r['expected_answer'])),parse('\\boxed{'+boxes[-1]+'}')):counts['invalid_or_unmatched_answer']+=1;continue
 n=len(tok.encode(text,add_special_tokens=False))+1
 if n+len(chat_prompt_ids(tok,r['problem']))>2048:counts['overlength']+=1;continue
 seen.add(key);kept.append(dict(problem=r['problem'],solution=text,expected_answer=r['expected_answer'],problem_source=r['problem_source'],source='nvidia/OpenMathInstruct-2',source_revision=revision,source_shard=meta['files'][0]['name'],stream_index=i,target_source='openmath405b_external',target_tokens=n,level='External',type=r['problem_source'],role='external_training_only'))
 if len(kept)%2000==0:print('Accepted',len(kept),'scanned',scanned,flush=True)
 if len(kept)>=16000:break
jsonl(out,kept);write(S/'audits/openmath_download.json',dict(time=time.time(),revision=revision,remote_file=url,remote_max_bytes=meta['files'][0]['size'],stream_shuffle_seed=20260929,shuffle_buffer=4096,scanned=scanned,accepted=len(kept),rejections=dict(counts),source_counts=dict(Counter(r['problem_source'] for r in kept)),sha256=sha(out),status='Provisional, requires near-duplicate screening, sampled process review and explicit release before training',limitation='Bounded shuffled stream from one fair-downsample shard, not a simple random sample of full14M. Synthetic expected answers may be majority-vote labels, not independently verified truth. No inference/evaluation results used.'))
print('Preparation complete',len(kept),flush=True)
