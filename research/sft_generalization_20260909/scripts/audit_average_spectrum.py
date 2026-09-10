"""Descriptive effective-update spectrum; never changes or evaluates a model."""
import torch
from safetensors.torch import load_file
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(4);results=[]
paths=[]
for parent in sorted((S/'results/training').glob('avg_teacher7*')):
 path=parent/'epoch4'
 if (path/'ready.json').exists():paths.append(path)
for path in paths:
 cfg=json.loads((path/'adapter_config.json').read_text());state=load_file(str(path/'adapter_model.safetensors'));scale=cfg['lora_alpha']/cfg['r'];modules=[]
 for key,A in state.items():
  if '.lora_A.' not in key:continue
  B=state[key.replace('.lora_A.','.lora_B.')]
  assert A.dtype==B.dtype==torch.float32 and A.shape[0]==B.shape[1]==cfg['r']
  _,rb=torch.linalg.qr(B.double(),mode='reduced');_,ra=torch.linalg.qr(A.double().T,mode='reduced')
  singular=torch.linalg.svdvals(rb@ra.T)*scale;energy=singular.square();total=float(energy.sum());assert total>0
  cumulative=energy.cumsum(0)/total
  ranks={str(threshold):int(torch.searchsorted(cumulative,torch.tensor(threshold,dtype=torch.float64)))+1 for threshold in [.9,.99,.999]}
  modules.append(dict(module=key,singular_values=singular.tolist(),squared_frobenius_norm=total,energy_rank=ranks,tail_energy_above_rank16=float(energy[16:].sum()),fraction_energy_above_rank16=float(energy[16:].sum()/total)))
 total=sum(r['squared_frobenius_norm'] for r in modules);tail=sum(r['tail_energy_above_rank16'] for r in modules)
 results.append(dict(name=json.loads((path/'ready.json').read_text())['name'],adapter_sha256=sha(path/'adapter_model.safetensors'),config_sha256=sha(path/'adapter_config.json'),inference_rank=cfg['r'],modules=modules,aggregate_squared_norm=total,aggregate_fraction_energy_above_per_module_rank16=tail/total,rank99_range=[min(r['energy_rank']['0.99'] for r in modules),max(r['energy_rank']['0.99'] for r in modules)]))
write(S/'audits/average_effective_update_spectrum.json',dict(time=time.time(),script_sha256=sha(__file__),models=results,scope='FP64 thin-QR/core-SVD spectrum of saved FP32 effective updates, no dense model mutation or inference. Concatenated averages have up to rank64 even though each source was trained rank16. Small tail energy does not prove rank16 compression preserves answers or rule out a capacity effect; no compression experiment performed.'))
print(json.dumps([dict(name=r['name'],energy_above_rank16=r['aggregate_fraction_energy_above_per_module_rank16'],rank99_range=r['rank99_range']) for r in results],indent=2))
