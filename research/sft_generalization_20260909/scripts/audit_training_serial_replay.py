"""Compare affected concurrent and dedicated-GPU training, retaining both artifacts."""
import torch
from safetensors.torch import load_file
from common import *
torch.set_num_threads(4)
plan=json.loads((S/'audits/affected_training_replay_plan.json').read_text())
original=S/'results/training'/plan['original'];replay=S/'results/training'/plan['replay']['name']
if not (original/'complete.json').exists() or not (replay/'complete.json').exists():print('Pending completed serial replay');raise SystemExit(0)
a=json.loads((original/'manifest.json').read_text());b=json.loads((replay/'manifest.json').read_text())
assert a['data_sha']==b['data_sha'] and a['script_sha']==b['script_sha']
aa={k:v for k,v in a['arguments'].items() if k!='name'};bb={k:v for k,v in b['arguments'].items() if k!='name'};assert aa==bb
exposure_equal=json.loads((original/'exposure.json').read_text())==json.loads((replay/'exposure.json').read_text());assert exposure_equal
comparisons=[]
for epoch in ['initial_adapter','epoch1','epoch2','epoch3','epoch4']:
 p=original/epoch/'adapter_model.safetensors';q=replay/epoch/'adapter_model.safetensors';x=load_file(str(p));y=load_file(str(q));assert set(x)==set(y)
 differences=[dict(tensor=k,max_abs=float((x[k]-y[k]).abs().max())) for k in x if not torch.equal(x[k],y[k])]
 comparisons.append(dict(checkpoint=epoch,all_tensors_exactly_equal=not differences,original_sha256=sha(p),serial_sha256=sha(q),different_tensor_count=len(differences),differences=differences))
all_equal=all(c['all_tensors_exactly_equal'] for c in comparisons)
write(S/'audits/affected_training_serial_replay_result.json',dict(time=time.time(),original=plan['original'],serial=plan['replay']['name'],same_config=True,same_source=True,same_data=True,exposure_sequence_equal=exposure_equal,all_checkpoints_bitwise_equal=all_equal,comparisons=comparisons,primary_for_future_coverage_analyses=plan['replay']['name'],interpretation='Dedicated-GPU serial replay is the primary coverage arm. Exact equality, if true, validates the measured checkpoints despite the disclosed handoff incident; it does not certify arbitrary concurrent GPU workloads. If different, retain incident results separately and use serial only.'))
print(json.dumps(dict(all_checkpoints_bitwise_equal=all_equal,checkpoints=[dict(checkpoint=c['checkpoint'],different_tensor_count=c['different_tensor_count']) for c in comparisons]),indent=2))
