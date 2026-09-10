"""Summarize online training diagnostics; no held-out accuracy inference."""
from common import *
runs={}
for name in ['teacher_all_kl0_lr5e5_s43','teacher_all_kl01_lr5e5_s43','teacher_all_kl1_lr5e5_s43']:
 p=S/'results/training'/name;assert (p/'complete.json').exists()
 h=json.loads((p/'history.json').read_text());steps=[r for r in h if 'train_token_loss' in r];ep={}
 for epoch in range(1,5):
  rows=[r for r in steps if r['epoch']==epoch];tokens=sum(r['supervised_tokens'] for r in rows)
  ce=sum(r['train_token_loss']*r['supervised_tokens'] for r in rows)/tokens
  kl=sum(r['forward_kl_per_token']*r['supervised_tokens'] for r in rows)/tokens if name!='teacher_all_kl0_lr5e5_s43' else None
  ep[str(epoch)]=dict(steps=len(rows),tokens=tokens,exposure_weighted_online_ce=ce,exposure_weighted_online_kl=kl,max_gradient_norm=max(r['gradient_norm'] for r in rows))
 runs[name]=dict(epochs=ep,history_sha256=sha(p/'history.json'),end_probe=[r for r in h if 'dev_ce' in r][-1])
write(S/'audits/anchoring_online_trajectory.json',dict(time=time.time(),script_sha256=sha(__file__),runs=runs,scope='Descriptive online losses at changing parameters on identical teacher-forced sequences. Lambda0 skips reference forward and has no measured KL, not zero KL. These measurements do not establish free-generation retention or a causal generalization gain.'))
print(json.dumps({n:r['epochs']['4'] for n,r in runs.items()},indent=2))
