"""Freeze new-target x objective controls; reuse completed CE arms."""
import ast
import contextlib
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from common import *

torch.set_num_threads(2)
source = S/'scripts/train.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'custom_backward')
namespace = dict(torch=torch, F=F)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
backward = namespace['custom_backward']
torch.manual_seed(20260909)
initial = torch.randn(3, 6, 11)
labels = torch.tensor([[-100,-100,2,3,4,5],[-100,1,2,-100,-100,-100],[-100,-100,-100,6,7,8]])
ids = torch.arange(3).view(3,1).expand(3,6)
class LogitModel:
 def __init__(self): self.logits=initial.clone().requires_grad_()
 def __call__(self,input_ids,attention_mask,use_cache): return SimpleNamespace(logits=self.logits[input_ids[:,0]])
def batch(ix): return dict(input_ids=ids[ix],attention_mask=torch.ones_like(ids[ix]),labels=labels[ix])
grads=[]; values=[]
with patch.object(torch.Tensor, 'cuda', lambda x: x), patch.object(torch, 'autocast', lambda *a,**k:contextlib.nullcontext()):
 for pieces in [[slice(0,3)],[slice(0,1),slice(1,3)]]:
  model=LogitModel(); value=backward(model,[batch(ix) for ix in pieces],'dft'); grads.append(model.logits.grad);values.append(value)
mask=labels[:,1:]!=-100
prob=initial[:,:-1].softmax(-1)
target=labels[:,1:].clamp_min(0)
onehot=F.one_hot(target,11)
weight=prob.gather(-1,target[...,None])
expected=weight*(prob-onehot)*mask[...,None]/mask.sum()
errors=[float((g[:,:-1]-expected).abs().max()) for g in grads]
assert max(errors)<1e-7
assert torch.allclose(grads[0],grads[1],atol=1e-7,rtol=1e-6)
assert all(torch.count_nonzero(g[:,-1])==0 for g in grads)
audit=dict(time=time.time(),train_source_sha256=sha(source),actual_function_ast_executed=True,
 cpu_only=True,cuda_and_autocast_mocked_only=True,analytical_max_abs_errors=errors,
 partition_max_abs_error=float((grads[0]-grads[1]).abs().max()),values=values,
 limitation='Checks shifted masked objective/stop-gradient/real-window denominator on controlled logits, not GPU model numerics or training outcome.')
write(S/'audits/dft_objective_gradient.json',audit)
path=S/'results/train_queue.json';queue=json.loads(path.read_text());byname={j['name']:j for j in queue}
pairs=[];new=[]
for prefix in ['sample_all','teacher32_all_qkvo16']:
 for lr in ['1e5','5e5']:
  control=byname[f'{prefix}_lr{lr}_s43'];job=dict(control)
  job.update(name=f'{prefix}_dft_lr{lr}_s43',objective='dft')
  assert (S/'results/training'/control['name']/'complete.json').exists()
  pairs.append(dict(control=control,treatment=job,data_sha256=sha(job['train_file'])))
  if job['name'] not in byname:new.append(job)
insert=next(i for i,j in enumerate(queue) if j['name']=='sample_all_all64_lr1e5_s43')
queue[insert:insert]=new
write(S/'audits/dft_interaction_plan.json',dict(time=time.time(),pairs=pairs,
 source='https://arxiv.org/html/2508.05629v3',training_script_unchanged=True,
 rationale='Prior raw-target DFT failure does not test interaction with accepted student/32B targets. Same data, initialization seed, exposure, optimizer and schedule as completed CE controls; objective only changes.',
 interpretation='Exploratory development only. DFT changes token gradient directions and Adam trajectories; not assumed equivalent to reducing LR. Any efficacy claim requires frozen multi-seed and held-out evaluation.',
 queue_total=len(queue)))
write(path,queue)
print(json.dumps(dict(gradient_audit=audit,added=[j['name'] for j in new],queue_total=len(queue))))
