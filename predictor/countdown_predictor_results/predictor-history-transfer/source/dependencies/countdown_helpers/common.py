import json, hashlib, re
from pathlib import Path
E=Path(__file__).resolve().parents[1]; R=E.parents[1]
MODEL='/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
TEACHER='/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
from gradient_geometry.extraction import SYSTEM_PROMPT, USER_TEMPLATE

def read(p):return [json.loads(l) for l in Path(p).read_text().split('\n') if l.strip()]
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows))
def key(p):return hashlib.sha256(re.sub(r'\s+','',p).encode()).hexdigest()
def prompt(tok,row):return tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':USER_TEMPLATE.format(problem=row['problem'])}],tokenize=False,add_generation_prompt=True)
def correct(pred,row):
 from math_verify import parse,verify
 return bool(verify(parse(row.get('gold',row['solution'])),parse(pred)))
