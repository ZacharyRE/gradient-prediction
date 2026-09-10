"""CPU gradient audit and prospective CE+KL retention-control release."""
import contextlib
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from common import *
from anchored_objective import anchored_backward,masked_forward_kl

torch.set_num_threads(2);torch.manual_seed(20261016)
initial=torch.randn(3,6,11);reference=torch.randn(3,6,11,requires_grad=True)
labels=torch.tensor([[-100,-100,2,3,4,5],[-100,1,2,-100,-100,-100],[-100,-100,-100,6,7,8]])
ids=torch.arange(3).view(3,1).expand(3,6);mask=labels[:,1:]!=-100;total=int(mask.sum())
class Model:
 def __init__(self):self.logits=initial.clone().requires_grad_();self.disabled=False
 @contextlib.contextmanager
 def disable_adapter(self):
  self.disabled=True
  try:yield
  finally:self.disabled=False
 def named_parameters(self):return [('lora_A',self.logits)]
 def __call__(self,input_ids,attention_mask,use_cache):return SimpleNamespace(logits=(reference if self.disabled else self.logits)[input_ids[:,0]])
def batch(ix):return dict(input_ids=ids[ix],attention_mask=torch.ones_like(ids[ix]),labels=labels[ix])
audits=[]
for objective in ['token_mean','dft']:
 grads=[];values=[]
 with patch.object(torch.Tensor,'cuda',lambda x:x),patch.object(torch,'autocast',lambda *a,**k:contextlib.nullcontext()):
  for pieces in [[slice(0,3)],[slice(0,1),slice(1,3)]]:
   model=Model();values.append(anchored_backward(model,[batch(i) for i in pieces],objective,.3,{'synthetic_nonzero_state':True}));grads.append(model.logits.grad)
 p=initial[:,:-1].softmax(-1);q=reference.detach()[:,:-1].softmax(-1);target=labels[:,1:].clamp_min(0);ce=p-F.one_hot(target,11)
 if objective=='dft':ce=ce*p.gather(-1,target[...,None])
 expected=(ce+.3*(p-q))*mask[...,None]/total
 errors=[float((g[:,:-1]-expected).abs().max()) for g in grads]
 assert max(errors)<1e-7 and torch.allclose(grads[0],grads[1],atol=1e-7,rtol=1e-6)
 assert reference.grad is None and all(torch.count_nonzero(g[:,-1])==0 for g in grads)
 audits.append(dict(objective=objective,analytic_max_abs_errors=errors,partition_max_abs_error=float((grads[0]-grads[1]).abs().max()),values=values,reference_gradient_absent=True))
 same=initial.clone().requires_grad_();zero=masked_forward_kl(same,initial,labels);zero.backward();assert float(zero)==0 and float(same.grad.abs().max())<1e-7
write(S/'audits/anchoring_objective_gradient.json',dict(time=time.time(),actual_helper_sha256=sha(S/'scripts/anchored_objective.py'),audits=audits,zero_at_reference=True,scope='Actual helper on controlled CPU logits, cuda/autocast mocked. Actual PEFT disable/restore and reference immutability are separately checked inside every nonzero-KL GPU training run.'))
queuepath=S/'results/train_queue.json';queue=json.loads(queuepath.read_text());control=next(j for j in queue if j['name']=='teacher_all_lr5e5_s43')
jobs=[]
for coefficient,label in [(0.,'0'),(.1,'01'),(1.,'1')]:
 job=dict(control,name=f'teacher_all_kl{label}_lr5e5_s43',trainer_script='train_anchored.py',kl_coefficient=coefficient);jobs.append(job)
assert not any(j['name'] in {q['name'] for q in queue} for j in jobs)
# Following the current-job drain, these explicit retention controls run next.
queue=jobs+queue
write(S/'audits/anchoring_plan.json',dict(time=time.time(),jobs=jobs,ce_control=control,data_sha256=sha(control['train_file']),original_train_sha256=sha(S/'scripts/train.py'),anchored_train_sha256=sha(S/'scripts/train_anchored.py'),helper_sha256=sha(S/'scripts/anchored_objective.py'),queue_total=len(queue),objective='Token-mean CE + lambda*token-mean full-vocabulary KL(base||adapted), on supervised target-prefix states including EOS.',reference='Same frozen BF16 base, PEFT adapter disabled only during no-grad reference forward; re-enabled before student forward/backward.',motivation='Measured fixed-seed dev gains on baseline-wrong questions are largely canceled by losses on baseline-correct questions. Tests a direct retention intervention.',limits=['KL states are training-target prefixes, not all baseline on-policy or heldout states.','Same LR/exposure/optimizer, but gradient trajectories change; no scalar-LR-equivalence claim.','Zero-coefficient full replay must match completed CE tensors/exposure before using nonzero-KL evidence.','Exploratory pilots; no new general efficacy claim before frozen multi-seed confirmation.'],literature='https://arxiv.org/html/2509.23753v3',literature_note='CE+KL control, not a full ASFT reproduction (ASFT uses DFT+KL).'))
write(queuepath,queue)
registry=S/'scripts/gpu_worker_registry.json';names=json.loads(registry.read_text());write(registry,sorted(set(names+['train_anchored.py','anchoring_restart_pipeline.py'])))
(S/'ANCHORING_READY').touch()
print(json.dumps(dict(gradient_audit=audits,queue_total=len(queue),jobs=jobs)))
