"""Per-completion teacher-forced NLL in the native BF16 SDPA-math arithmetic."""
import argparse,torch,subprocess,gc
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from common import *
from training.train_lora_sft import CompletionDataset,Collator
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--batch',type=int,default=8);p.add_argument('--judge-after',type=Path);p.add_argument('--sampling-after',type=Path);a=p.parse_args();gpu_guard()
 rows=read(a.input);out=S/'results/candidate_nll'/a.name;out.mkdir(parents=True,exist_ok=False)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);ds=CompletionDataset(a.input,tok,2048);assert ds.truncated==0 and len(ds)==len(rows)
 collator=Collator(tok.pad_token_id);torch.set_num_threads(4);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();model.config.use_cache=False
 write(out/'manifest.json',dict(time=time.time(),model=MODEL,input=str(a.input),input_sha256=sha(a.input),script_sha256=sha(__file__),dtype='bfloat16',attention='explicit_sdpa_math',batch=a.batch,include_eos=True,loss='Mean over shifted completion+EOS tokens; prompt/pad labels ignored',n=len(rows),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 (out/'score_source.py').write_text(Path(__file__).read_text())
 for start in range(0,len(ds),a.batch):
  gpu_guard();batch={k:v.cuda() for k,v in collator([ds[i] for i in range(start,min(start+a.batch,len(ds)))]).items()}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits[:,:-1,:].float();labels=batch['labels'][:,1:]
   loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').reshape(labels.shape)
   count=(labels!=-100).sum(1);sums=loss.sum(1);means=sums/count
  assert torch.isfinite(means).all() and (count>0).all()
  with (out/'scores.jsonl').open('a') as f:
   for j in range(len(means)):
    r=rows[start+j];f.write(json.dumps(dict(index=start+j,original_index=r['original_index'],target_sha256=r['target_sha256'],candidate_sources=r['candidate_sources'],tokens=int(count[j]),nll_sum=float(sums[j]),nll_mean=float(means[j])))+'\n')
  del logits,labels,loss,batch
  if start%(a.batch*50)==0:print(json.dumps(dict(completed=min(start+a.batch,len(ds)),total=len(ds),time=time.time())),flush=True)
 write(out/'complete.json',dict(n=len(rows),time=time.time()))
 if a.judge_after or a.sampling_after:
  del model;gc.collect();torch.cuda.empty_cache();gpu_guard()
 if a.judge_after:
  command=[sys.executable,str(S/'scripts/judge_targets.py'),'--input',str(a.judge_after),'--name','guided_legacy_retry']
  with (S/'logs/judge_guided_retry.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits/judge_guided_retry_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time(),gpu=1,reason='Initial judge startup failed before predictions because virtualenv bin was missing from PATH; retry with executable directory on PATH. Native diagnostic already runs; serialize retry after candidate scoring.'))
 if a.sampling_after:
  gpu_guard();command=[sys.executable,str(S/'scripts/sampling_probe.py'),'--plan',str(a.sampling_after)]
  with (S/'logs/sampling_probe.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
  write(S/'audits/sampling_probe_dispatch.json',dict(command=command,returncode=r.returncode,time=time.time(),gpu=1))
if __name__=='__main__':main()
