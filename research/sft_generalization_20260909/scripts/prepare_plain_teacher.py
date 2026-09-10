"""Conservative markup-only32B target control, preserving mathematical spans and words."""
import re
from transformers import AutoTokenizer
from common import *
from training.train_lora_sft import chat_prompt_ids
protected=re.compile(r'(```.*?```|\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]|\\\(.*?\\\))',re.S)
def clean(text):
 parts=protected.split(text)
 for i in range(0,len(parts),2):
  x=re.sub(r'^#{1,6}[ \t]+','',parts[i],flags=re.M)
  x=re.sub(r'(?<![\w$\\])\*\*(?=\S)([^*\n]+?)\*\*(?!\w)',r'\1',x)
  x=re.sub(r'^[ \t]*---+[ \t]*$','',x,flags=re.M);parts[i]=x
 return ''.join(parts)
rows=read(S/'data/teacher32_all.jsonl');tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);new=[];audit=[]
for r in rows:
 old=r['solution'];text=clean(old);assert protected.findall(old)==protected.findall(text)
 assert re.findall(r'[A-Za-z0-9_\\]+',old)==re.findall(r'[A-Za-z0-9_\\]+',text)
 n=len(tok.encode(text,add_special_tokens=False))+1;assert n+len(chat_prompt_ids(tok,r['problem']))<=2048
 rr=dict(r,solution=text,target_source='teacher32_plain_markup',target_tokens=n,target_sha256=hashlib.sha256(text.encode()).hexdigest(),candidate_sources=['teacher32_plain_markup']);new.append(rr);audit.append(dict(original_index=r['original_index'],old_sha256=hashlib.sha256(old.encode()).hexdigest(),new_sha256=rr['target_sha256'],old_tokens=r['target_tokens'],new_tokens=n))
jsonl(S/'data/teacher32_plain_markup.jsonl',new);jsonl(S/'audits/plain_teacher_transform.jsonl',audit)
write(S/'audits/plain_teacher_plan.json',dict(time=time.time(),n=len(new),data_sha256=sha(S/'data/teacher32_plain_markup.jsonl'),source_sha256=sha(S/'data/teacher32_all.jsonl'),old_tokens=sum(r['target_tokens'] for r in rows),new_tokens=sum(r['target_tokens'] for r in new),changed=sum(r['old_sha256']!=r['new_sha256'] for r in audit),protected_math_code_byte_identical=True,all_word_identifier_numeric_tokens_identical=True,purpose='Test markup-only component of style mismatch; same questions, wording, mathematics and known target errors. Does not shorten explanations or repair reasoning.',status='PreparedNLLdiagnostic; no training queued yet. Any later training must record paired-control release.'))
p=S/'audits/sampling_seed_stability_plan.json';plan=json.loads(p.read_text());plan['after_sampling_tasks'].insert(1,dict(name='plain_teacher_nll',command=[sys.executable,str(S/'scripts/score_candidates.py'),'--input',str(S/'data/teacher32_plain_markup.jsonl'),'--name','teacher32_plain_markup'],log=str(S/'logs/plain_teacher_nll.log')));write(p,plan)
print('Prepared',len(new),'targets; tokens',sum(r['target_tokens'] for r in rows),'->',sum(r['target_tokens'] for r in new))
