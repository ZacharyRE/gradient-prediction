"""Fixed-endpoint development retention/rescue accounting, conditional on baseline."""
import numpy as np
from common import *
root=S/'results/evaluation/dev';base=read(root/'base/math_predictions.jsonl');b=np.array([r['correct'] for r in base],int)
assert json.loads((S/'audits/sampling_restart_protocol_replay.json').read_text())['passed']
arms=[
 ('teacher_all_kl01_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4','KL0.1'),
 ('teacher_all_kl1_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4','KL1'),
 ('sample_keepgreedy_lr1e5_s43_epoch2','sample_all_lr1e5_s43_epoch2','student_keepgreedy'),
 ('teacher_keepgreedy_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4','teacher_keepgreedy'),
]
def load(name):
 p=root/name/'math_predictions.jsonl';r=read(p)
 assert [x['sample_hash'] for x in r]==[x['sample_hash'] for x in base]
 return np.array([x['correct'] for x in r],int)
def summary(d):
 rng=np.random.default_rng(20260923);boots=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*100
 return dict(n=len(d),delta_pp=float(d.mean()*100),paired_question_ci95_pp=np.quantile(boots,[.025,.975]).tolist(),control_wrong_to_treatment_right=int((d>0).sum()),control_right_to_treatment_wrong=int((d<0).sum()))
result=[];pending=[]
for treatment,control,label in arms:
 if not all((root/name/'math_summary.json').exists() for name in [treatment,control]):pending.append(label);continue
 if label.startswith('KL'):
  assert json.loads((S/'audits/anchor_zero_replay_result.json').read_text())['passed']
  ref=json.loads((S/'results/training'/treatment.rsplit('_epoch',1)[0]/'anchoring_audit.json').read_text())
  assert ref['reference_unchanged'] and ref['first_batch']['initial_logits_max_abs']==0
 t=load(treatment);c=load(control);d=t-c
 result.append(dict(label=label,treatment=treatment,control=control,treatment_correct=int(t.sum()),control_correct=int(c.sum()),treatment_baseline_wrong_rescued=int(t[b==0].sum()),control_baseline_wrong_rescued=int(c[b==0].sum()),treatment_baseline_correct_lost=int((1-t[b==1]).sum()),control_baseline_correct_lost=int((1-c[b==1]).sum()),all_questions=summary(d),baseline_wrong_questions=summary(d[b==0]),baseline_correct_questions=summary(d[b==1]),predictions_sha256={name:sha(root/name/'math_predictions.jsonl') for name in [treatment,control]}))
write(S/'results/retention_control_analysis.json',dict(time=time.time(),script_sha256=sha(__file__),baseline_correct=int(b.sum()),baseline_wrong=int((1-b).sum()),comparisons=result,pending=pending,scope='Fixed-endpoint seed43 exploratory controls on development500. Positive conditional delta means more retained/rescued correct answers. Conditional intervals are unadjusted and do not establish multiple-seed or fresh general efficacy; the two baseline strata together give the net effect.'))
print(json.dumps(result,indent=2))
