"""Development-only paired contrasts, preserving treatment/control semantics."""
import numpy as np
from scipy.stats import binomtest
from common import *
root=S/'results/evaluation/dev';pairs=[]
def add(treatment,control,hypothesis,epochs=range(1,5)):
 for e in epochs:pairs.append((treatment+f'_epoch{e}',control+f'_epoch{e}',hypothesis))
for lr in ['1e5','5e5']:
 add(f'teacher_all_qkvo64_lr{lr}_s43',f'teacher_all_lr{lr}_s43','General7B same-source QKVO rank64 vs16; alpha/r=1 and same initial zero function, A shapes differ; fixed primaryepoch4')
 for source in ['sample_all','teacher32_all_qkvo16']:
  add(f'{source}_dft_lr{lr}_s43',f'{source}_lr{lr}_s43','Detached token-probability weighted DFT vs CE on identical new targets, initialization seed, exposure and optimizer; objective-only intervention, not assumed scalar LR equivalence')
 add(f'teacher_common_lr{lr}_s43',f'sample_common_lr{lr}_s43','7B vs student targets, same1429 questions/order; nearly equal target tokens')
 for source in ['sample','teacher']:
  add(f'{source}_all_all16_lr{lr}_s43',f'{source}_all_lr{lr}_s43','All-linear vs QKVO LoRA, same targets and exact shared attention initialization')
 add(f'sample_augmented_lr{lr}_s43',f'sample_augmented_rawhard_lr{lr}_s43','Sampled vs raw targets on added questions, exact question/order match')
 add(f'sample_augmented_lr{lr}_s43',f'sample_greedy_budget_lr{lr}_s43','Added question coverage vs old-example repetition, equal number of slots but different question/token distributions')
 add(f'sample_multi_lr{lr}_s43',f'sample_repeat_lr{lr}_s43','Multiple distinct answer texts vs repeat, exact4-slot question/order match',range(1,2))
 add(f'teacher32_all_all16_lr{lr}_s43',f'teacher32_all_qkvo16_lr{lr}_s43','All-linear vs QKVO on32B targets, exact shared attention initialization')
 add(f'teacher32_plain_markup_lr{lr}_s43',f'teacher32_all_qkvo16_lr{lr}_s43','Cosmetic markdown removal only; same1637 questions/order and protected math/word/number content; known original process errors preserved in both arms')
 add(f'nll_noguided_min_lr{lr}_s43',f'nll_noguided_random_lr{lr}_s43','Within-question minimum student NLL vs random valid target, identical1758 questions/order; about1.1percent target-token difference; guided targets excluded')
 add(f'expansion32_combined_lr{lr}_s43',f'expansion32_repeat_control_lr{lr}_s43','Broader human-question coverage vs repeating cleaned old core; identical4353 slots/horizon, about1.8percent target-token difference')
 add(f'sample_expansion_combined_lr{lr}_s43',f'sample_expansion_repeat_control_lr{lr}_s43','Broader human-question coverage with sampled student targets vs repeating cleaned1561-question core; same4149 slots/horizon, about1.46percent target-token difference')
 for source,rank16 in [('sample','sample_all_all16'),('teacher32','teacher32_all_all16')]:
  add(f'{source}_all_all64_lr{lr}_s43',f'{rank16}_lr{lr}_s43','Rank64 vs rank16 all-linear; same targets and alpha/r=1, but different-shaped random A factors, not bitwise-identical parameter initialization')
add('expansion32_combined_lr1e5_s43','expansion32_rawnew_control_lr1e5_s43','32B vs raw reference on exactly the same added questions, unchanged cleaned old core; substantial target-length difference')
add('sample_expansion_combined_lr1e5_s43','sample_expansion_rawnew_control_lr1e5_s43','Student sampled vs raw targets on same2588 added questions, same1561 old-core targets and4149 positions; target-token totals1952239 vs1194784')
add('sample_expansion_common_teacher32_lr1e5_s43','teacher32_expansion_common_sample_lr1e5_s43','Student vs32B target source on exact3897 old/new shared questions/order; target-token totals1778132 vs1881235')
add('teacher7_expansion_combined_lr5e5_s43','teacher7_expansion_repeat_control_lr5e5_s43','Broader human-question coverage with general7B targets vs repeating cleaned1542-question core; same4145 slots/horizon, target-token totals1989823 vs1974549; primary fixedepoch4')
add('teacher7_expansion_combined_lr5e5_s43','teacher7_expansion_rawnew_control_lr5e5_s43','General7B vs raw targets on same2603 added questions, identical1542 old-core targets and4145 positions; target-token totals1989823 vs1189608; primary fixedepoch4')
add('teacher32_common_teacher7_lr1e5_s43','teacher7_common_teacher32_lr1e5_s43','32B vs7B targets, exact1465 questions/order; different target lengths and teacher family')
add('teacher_math7_common_teacher7_lr1e5_s43','teacher7_common_math7_lr1e5_s43','Math-specialized7B with official CoT teacher prompt vs general7B target pipeline; same1466 questions/order after known-error exclusions, target-token totals724811 vs675951; does not isolate teacher checkpoint from teacher prompt')
add('sample_keepgreedy_lr1e5_s43','sample_all_lr1e5_s43','Preserve prior greedy-correct target text on same1563 questions/order; all396 originally wrong-question targets unchanged. Primary matched endpoint epoch2; changed correct-target content and length are joint intervention',range(2,3))
add('teacher_keepgreedy_lr5e5_s43','teacher_all_lr5e5_s43','Preserve prior greedy-correct target text on same1547 questions/order; all401 originally wrong-question targets unchanged. Primary matched endpoint epoch4; changed correct-target content and length are joint intervention',range(4,5))
add('sample_all_lr1e5_s43','sample_matchedraw_lr1e5_s43','Student sampled vs original reference targets on identical1563 questions/order at fixedepoch2, same initialization/exposure/optimizer. Content and length jointly change; conditional target-pipeline effect',range(2,3))
add('teacher_all_lr5e5_s43','teacher_matchedraw_lr5e5_s43','General7B vs original reference targets on identical1547 questions/order at fixedepoch4, same initialization/exposure/optimizer. Content and length jointly change; conditional target-pipeline effect',range(4,5))
for coefficient in ['01','1']:
 add(f'teacher_all_kl{coefficient}_lr5e5_s43','teacher_all_lr5e5_s43','Forward KL(base||adapted) retention penalty added to CE on identical7B targets/exposure/seed/optimizer; separate trainer lambda0 replay and per-run reference audits required')
add('sample_all_shortcos2_lr1e5_s43','sample_all_lr1e5_s43','Completed2-epoch cosine vs8-epoch horizon at identical2-epoch exposure; optimization trajectory differs',range(2,3))
add('teacher_all_shortcos4_lr5e5_s43','teacher_all_lr5e5_s43','Completed4-epoch cosine vs8-epoch horizon at identical4-epoch exposure; optimization trajectory differs',range(4,5))
add('teacher_all_dftkl0_lr5e5_s43','teacher_all_lr5e5_s43','DFT vs CE on identical general7B targets, seed43, LR5e-5 and exposure; fixed primaryepoch4')
add('teacher_all_dftkl1_lr5e5_s43','teacher_all_dftkl0_lr5e5_s43','ForwardKL1 added to DFT on identical general7B targets and optimizer; ASFT-inspired objective control, not paper-scale replication; fixed primaryepoch4')
add('teacher_all_dftkl01_lr5e5_s43','teacher_all_dftkl0_lr5e5_s43','ForwardKL0.1 added to DFT on identical general7B targets and optimizer; coefficient companion predeclared before any DFT training/scores',range(4,5))
for coefficient in ['01','1']:
 add(f'teacher_all_dftkl{coefficient}_lr5e5_s43',f'teacher_all_kl{coefficient}_lr5e5_s43','DFT vs CE with identical forwardKL coefficient and general7B target data; fixed primaryepoch4',range(4,5))
for source in ['teacher7_expansion_combined','teacher7_expansion_repeat_control','teacher7_expansion_rawnew_control']:
 add('avg_'+source+'_1234_s43',source+'_lr5e5_s43','Exact effective-update epoch1..4 averaging vs own epoch4 for7B expansion arm, prospectively declared before these trainings',range(4,5))
add('avg_teacher7_expansion_combined_1234_s43','avg_teacher7_expansion_repeat_control_1234_s43','Both use prospectively declared epoch1..4 averages; expanded question coverage vs old-core repetition with identical slots/horizon',range(4,5))
add('avg_teacher7_expansion_combined_1234_s43','avg_teacher7_expansion_rawnew_control_1234_s43','Both use epoch1..4 averages; generated vs raw new targets on samequestions/order with identical old core',range(4,5))
out=[]
for t,c,h in pairs:
 if t.startswith(('teacher_all_dftkl1_','teacher_all_dftkl01_')):
  run_audit=S/'results/training'/t.rsplit('_epoch',1)[0]/'anchoring_audit.json'
  if not run_audit.exists():continue
  ra=json.loads(run_audit.read_text());assert ra['reference_unchanged'] and ra['first_batch']['initial_logits_max_abs']==0
 if t.startswith(('teacher_all_kl01_','teacher_all_kl1_')):
  replay_audit=S/'audits/anchor_zero_replay_result.json';run_audit=S/'results/training'/t.rsplit('_epoch',1)[0]/'anchoring_audit.json'
  if not replay_audit.exists() or not run_audit.exists():continue
  assert json.loads(replay_audit.read_text())['passed']
  ra=json.loads(run_audit.read_text());assert ra['reference_unchanged'] and ra['first_batch']['initial_logits_max_abs']==0
 if t.startswith('sample_augmented_lr1e5_s43_epoch'):
  t=t.replace('sample_augmented_lr1e5_s43_epoch','sample_augmented_lr1e5_s43_serial_replay_epoch')
  h+='; dedicated-GPU serial replay is primary after disclosed handoff incident'
 tf=root/t/'math_summary.json';cf=root/c/'math_summary.json'
 if not tf.exists() or not cf.exists():continue
 tr=read(tf.with_name('math_predictions.jsonl'));cr=read(cf.with_name('math_predictions.jsonl'))
 assert [r['sample_hash'] for r in tr]==[r['sample_hash'] for r in cr]
 x=np.array([r['correct'] for r in tr],int);y=np.array([r['correct'] for r in cr],int);d=x-y
 rng=np.random.default_rng(20260909);boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)*100
 w=int(((x==1)&(y==0)).sum());l=int(((x==0)&(y==1)).sum())
 out.append(dict(treatment=t,control=c,hypothesis=h,n=len(d),treatment_correct=int(x.sum()),control_correct=int(y.sum()),delta_pp=100*d.mean(),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),wins=w,losses=l,p=binomtest(w,w+l).pvalue if w+l else 1))
prev=0
for i,r in enumerate(sorted(out,key=lambda r:r['p'])):prev=max(prev,min(1.,r['p']*(len(out)-i)));r['holm_p']=prev
write(S/'results/development_pairs.json',dict(time=time.time(),status='Exploratory; does not turn checkpoint selection into confirmatory evidence',comparisons=out))
print(json.dumps(dict(complete=len(out),positive_ci=[r for r in out if r['ci95_pp'][0]>0]),indent=2))
