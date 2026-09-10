"""Average effective LoRA updates by exact factor concatenation, never average A/B separately."""
import torch
from safetensors.torch import load_file,save_file
from common import *
torch.set_num_threads(4)
plan=json.loads((S/'audits/epoch_average_plan.json').read_text())
for job in plan['jobs']:
 dest=S/'results/training'/job['name']/f"epoch{job['endpoint']}"
 if (dest/'ready.json').exists():continue
 sources=[Path(x) for x in job['sources']]
 if not all((x/'ready.json').exists() for x in sources):continue
 assert not dest.exists();configs=[json.loads((x/'adapter_config.json').read_text()) for x in sources];cfg=configs[0]
 assert all(c==cfg for c in configs)
 for k in ['use_dora','use_rslora','rank_pattern','alpha_pattern','modules_to_save']:assert not cfg.get(k)
 assert cfg.get('bias','none')=='none'
 states=[load_file(str(x/'adapter_model.safetensors')) for x in sources];assert all(set(s)==set(states[0]) for s in states)
 n=len(states);scale=cfg['lora_alpha']/cfg['r'];rank=n*cfg['r'];assert rank<=64 and n in [2,4];combined={};checks=[];g=torch.Generator().manual_seed(20261003)
 for k in states[0]:
  if '.lora_A.' not in k:continue
  bk=k.replace('.lora_A.','.lora_B.');aa=[s[k] for s in states];bb=[s[bk] for s in states];assert all(x.dtype==torch.float32 for x in aa+bb)
  A=torch.cat(aa,dim=0);B=torch.cat([b*(scale/n) for b in bb],dim=1);combined[k]=A;combined[bk]=B
  z=torch.randn(A.shape[1],7,generator=g,dtype=torch.float64);expected=sum(b.double()@(a.double()@z)*(scale/n) for a,b in zip(aa,bb));actual=B.double()@(A.double()@z);err=float((actual-expected).abs().max());den=float(expected.abs().max());assert err<1e-10*max(1,den);checks.append(dict(module=k,max_abs_action_error=err,reference_max_abs=den))
 assert set(combined)==set(states[0]);dest.mkdir(parents=True);save_file(combined,str(dest/'adapter_model.safetensors'));newcfg=dict(cfg,r=rank,lora_alpha=rank);write(dest/'adapter_config.json',newcfg)
 manifest=dict(kind='derived_epoch_weight_average_not_new_training',time=time.time(),sources=[dict(path=str(p),weight_sha256=sha(p/'adapter_model.safetensors'),config_sha256=sha(p/'adapter_config.json')) for p in sources],formula='A_concat=vertical_stack(A_i), B_concat=horizontal_stack((alpha_i/r_i)/N*B_i), alpha_new/r_new=1; effective delta equals mean delta_i in real arithmetic.',training_rank=cfg['r'],inference_rank=rank,not_prediction_ensemble=True,checks=checks,script_sha256=sha(__file__),limitation='Exact mathematical weight average with FP32 factors and FP64 action checks; BF16 unmerged runtime numerics need not equal averaging merged weights or predictions.')
 write(dest/'derivation.json',manifest);write(dest.parent/'manifest.json',manifest);write(dest/'ready.json',dict(name=job['name']+f"_epoch{job['endpoint']}",adapter=str(dest.resolve()),epoch=job['endpoint'],derivation=manifest,time=time.time()));print('Created',job['name'],flush=True)
