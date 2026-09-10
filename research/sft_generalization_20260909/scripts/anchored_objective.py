"""Token-mean CE/DFT plus forward KL from the frozen, adapter-disabled base."""
import hashlib
import torch
import torch.nn.functional as F

def masked_forward_kl(student_logits, reference_logits, labels):
 mask=labels[:,1:]!=-100
 logp=F.log_softmax(student_logits[:,:-1][mask].float(),dim=-1)
 logq=F.log_softmax(reference_logits[:,:-1][mask].detach().float(),dim=-1)
 return (logq.exp()*(logq-logp)).sum()

def anchored_backward(model,window,objective,coefficient,audit):
 counts=[int((b['labels'][:,1:]!=-100).sum()) for b in window];total=sum(counts)
 ce=weighted=kl_total=value=0.
 for batch in window:
  b={k:v.cuda() for k,v in batch.items()}
  with model.disable_adapter(),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   reference=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits.detach()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits
  labels=b['labels'][:,1:];mask=labels!=-100
  losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='none').view_as(labels)
  fit=losses.sum() if objective=='token_mean' else (losses*torch.exp(-losses).detach()*mask).sum()
  kl=masked_forward_kl(logits,reference,b['labels'])
  if not audit:
   audit.update(initial_logits_max_abs=float((logits.detach()-reference).abs().max()),initial_kl_per_token=float(kl.detach())/int(mask.sum()),reference_requires_grad=reference.requires_grad,trainable_adapter_parameters_restored=all(p.requires_grad for n,p in model.named_parameters() if 'lora_' in n))
   assert audit['initial_logits_max_abs']==0 and abs(audit['initial_kl_per_token'])<1e-7
   assert not audit['reference_requires_grad'] and audit['trainable_adapter_parameters_restored']
  loss=(fit+coefficient*kl)/total
  assert torch.isfinite(loss) and float(kl.detach())/int(mask.sum())>-1e-6
  loss.backward();ce+=float(losses.detach().sum());weighted+=float(fit.detach());kl_total+=float(kl.detach());value+=float(loss.detach())
 return dict(train_token_loss=ce/total,train_objective_loss=value,fit_objective_loss=weighted/total,forward_kl_per_token=kl_total/total,kl_coefficient=coefficient,supervised_tokens=total,microbatch_supervised_tokens=counts)

def reference_signature(model,batch):
 was_training=model.training;model.eval();b={k:v.cuda() for k,v in batch.items()}
 with model.disable_adapter(),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  logits=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits
  chosen=logits[:,[0,logits.shape[1]//2,logits.shape[1]-1],:].detach().contiguous().cpu()
 signature=hashlib.sha256(chosen.view(torch.uint16).numpy().tobytes()).hexdigest()
 model.train(was_training)
 return signature
