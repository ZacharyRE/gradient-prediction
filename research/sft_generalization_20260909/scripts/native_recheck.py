"""New-target adapters under the actual HF/PEFT SDPA-math training arithmetic."""
import argparse
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
from common import *
from evaluation.evaluate_sft_math_benchmarks import make_prompt,is_correct
from gradient_geometry.sft_protocol import artifact_identity,sample_identity
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--limit',type=int,default=128);p.add_argument('--batch',type=int,default=8);a=p.parse_args();gpu_guard()
 rows=read(S/'data/dev.jsonl')[:a.limit];items=json.loads(a.manifest.read_text());torch.set_num_threads(4)
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
 base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
 for name,adapter in items.items():
  out=S/'results/native/dev'/name
  if (out/'summary.json').exists():continue
  out.mkdir(parents=True,exist_ok=False);gpu_guard();model=base;loaded_check=None
  if adapter:
   model=PeftModel.from_pretrained(base,adapter,autocast_adapter_dtype=True)
   actual=get_peft_model_state_dict(model);expected=load_file(str(Path(adapter)/'adapter_model.safetensors'))
   loaded_check=all(torch.equal(actual[k].detach().cpu(),v) for k,v in expected.items());assert loaded_check
   assert all(v.dtype==torch.float32 for k,v in model.named_parameters() if 'lora_' in k)
  model.eval();write(out/'manifest.json',dict(model=artifact_identity(Path(MODEL)),adapter=artifact_identity(Path(adapter)) if adapter else None,adapter_loaded_exactly=loaded_check,script_sha256=sha(__file__),data_sha256=sha(S/'data/dev.jsonl'),selected_first_n=len(rows),batch=a.batch,attention='explicit_sdpa_math',base_dtype='bfloat16',adapter_dtype='float32',compute_autocast='bfloat16',use_cache=True,temperature=0,max_new_tokens=2048,start=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
  predictions=[];start=time.time()
  for offset in range(0,len(rows),a.batch):
   gpu_guard();chunk=rows[offset:offset+a.batch];batch=tok([make_prompt(tok,r['problem']) for r in chunk],padding=True,return_tensors='pt').to('cuda')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    generated=model.generate(**batch,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,max_new_tokens=2048,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
   for row,ids in zip(chunk,generated[:,batch['input_ids'].shape[1]:]):
    values=ids.tolist();stopped=tok.eos_token_id in values;end=values.index(tok.eos_token_id) if stopped else len(values);prediction=tok.decode(values[:end],skip_special_tokens=True)
    predictions.append(dict(index=len(predictions),sample_hash=sample_identity(row),prediction=prediction,correct=is_correct('math',row,prediction),generated_tokens=end+int(stopped),finish_reason='stop' if stopped else 'length'))
   jsonl(out/'predictions.jsonl',predictions);print(json.dumps(dict(name=name,completed=len(predictions),correct=sum(r['correct'] for r in predictions),elapsed=time.time()-start)),flush=True)
  write(out/'summary.json',dict(n=len(rows),correct=sum(r['correct'] for r in predictions),elapsed=time.time()-start))
  if adapter:base=model.unload();del model
  torch.cuda.empty_cache()
if __name__=='__main__':main()
