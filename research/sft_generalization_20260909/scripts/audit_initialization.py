"""Audit exact zero-function and same-setting initialization of existing runs."""
from collections import defaultdict
import torch
from safetensors.torch import load_file
from common import *
groups=defaultdict(list);details=[]
reference=S/'results/training/greedy_lr5e5_s43/initial_adapter/adapter_model.safetensors';attention=load_file(str(reference));assert all(torch.count_nonzero(v)==0 for k,v in attention.items() if '.lora_B.' in k)
for m in sorted((S/'results/training').glob('*/manifest.json')):
 manifest=json.loads(m.read_text())
 if 'arguments' not in manifest or manifest.get('resumed_from'):continue
 a=manifest['arguments'];p=m.parent/'initial_adapter/adapter_model.safetensors'
 if not p.exists():continue
 state=load_file(str(p));zero=all(torch.count_nonzero(v)==0 for k,v in state.items() if '.lora_B.' in k);assert zero
 key=(a['seed'],a['modules'],a['rank'],a['alpha']);digest=sha(p);groups[key].append(dict(name=m.parent.name,sha256=digest))
 shared=None
 if a['seed']==43 and a['rank']==16:
  shared=all(k in state and torch.equal(v,state[k]) for k,v in attention.items());assert shared
 details.append(dict(name=m.parent.name,group=key,n_tensors=len(state),all_B_zero=True,shared_reference_attention_exact=shared,parameter_dtypes=sorted({str(v.dtype) for v in state.values()})))
group_rows=[]
for key,rows in groups.items():
 same=len({r['sha256'] for r in rows})==1;assert same
 group_rows.append(dict(key=key,all_initial_weight_files_equal=same,runs=rows))
write(S/'audits/initialization_all_current.json',dict(time=time.time(),reference=str(reference),groups=group_rows,details=details,limitation='Rank/seed groups differ in initialization. Zero B gives identical mathematical initial update, not an assertion about arbitrary inference kernels.'))
print('Audited',len(details),'runs in',len(groups),'groups')
