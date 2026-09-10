"""Download one pinned domain-specific7B teacher into the study's own directory."""
from huggingface_hub import HfApi,snapshot_download
from common import *
repo='Qwen/Qwen2.5-Math-7B-Instruct';info=HfApi().model_info(repo,files_metadata=True);allow=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','LICENSE','README.md'];size=sum(x.size or 0 for x in info.siblings if x.rfilename.endswith('.safetensors'));assert size<17_000_000_000
out=S/'models/Qwen2.5-Math-7B-Instruct';write(S/'audits/math_teacher_download_plan.json',dict(time=time.time(),repo=repo,revision=info.sha,weight_bytes=size,local_dir=str(out),purpose='Same-family math-specialized teacher hypothesis; student remains existingQwen2.5-1.5B-Instruct. No claim based on official score alone; no GPU used for download.'))
path=snapshot_download(repo_id=repo,revision=info.sha,allow_patterns=allow,local_dir=out,max_workers=4)
index=json.loads((out/'model.safetensors.index.json').read_text());assert all((out/f).exists() for f in set(index['weight_map'].values()));write(S/'audits/math_teacher_download_complete.json',dict(time=time.time(),path=path,revision=info.sha,files=[dict(name=p.name,size=p.stat().st_size) for p in out.iterdir() if p.is_file()]));print('Math teacher download complete',flush=True)
