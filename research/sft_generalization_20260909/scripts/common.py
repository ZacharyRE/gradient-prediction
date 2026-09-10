import hashlib,json,os,sys,time,tempfile
from pathlib import Path
S=Path(__file__).resolve().parents[1]
ROOT=S.parents[1]
OLD=ROOT/'research/sft_diagnosis_20260908'
FOLLOW=OLD/'causal_followup'
MODEL='/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
TEACHER='/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
DEADLINE=1789053400
sys.path.insert(0,str(OLD/'snapshots'))
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=p.parent,prefix=p.name+'.',suffix='.tmp',delete=False) as f:
  json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n');temp=Path(f.name)
 temp.replace(p)
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def gpu_guard():
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','3'],'Only one of GPU1/GPU3 per worker'
 assert time.time()<DEADLINE,'24-hour deadline reached'
