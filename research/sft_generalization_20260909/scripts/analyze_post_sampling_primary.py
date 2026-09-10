"""Consolidate fixed post-diagnostic endpoints and strict scoring sensitivity."""
import numpy as np
from common import *
root=S/'results/evaluation/dev';audit_path=S/'audits/scoring_post_sampling_primary.json';audit=json.loads(audit_path.read_text())
assert audit['dataset_sha256']==sha(S/'data/dev.jsonl')
assert json.loads((S/'audits/sampling_restart_protocol_replay.json').read_text())['passed']
raw={};strict={};base=read(root/'base/math_predictions.jsonl')
for name,record in audit['models'].items():
 p=root/name/'math_predictions.jsonl';rr=read(p);assert sha(p)==record['predictions_sha256']
 assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base]
 raw[name]=np.array([r['correct'] for r in rr],int);strict[name]=np.array(record['strict_vector'],int)
 assert raw[name].shape==strict[name].shape==(500,)
def summarize(d):
 rng=np.random.default_rng(20261021);qb=[];sb=[]
 for _ in range(10000):
  ix=rng.integers(d.shape[1],size=d.shape[1]);si=rng.integers(d.shape[0],size=d.shape[0]);qb.append(d[:,ix].mean()*100);sb.append(d[si][:,ix].mean()*100)
 return dict(mean_delta_pp=float(d.mean()*100),per_seed_delta_pp=(d.mean(1)*100).tolist(),all_seeds_positive=bool((d.mean(1)>0).all()),question_ci95_pp=np.quantile(qb,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sb,[.025,.975]).tolist())
pairs=[
 ('student_source','sample_all_lr1e5_s43_epoch2','sample_matchedraw_lr1e5_s43_epoch2'),
 ('teacher_source','teacher_all_lr5e5_s43_epoch4','teacher_matchedraw_lr5e5_s43_epoch4'),
 ('coverage_vs_repeat','sample_augmented_lr1e5_s43_serial_replay_epoch2','sample_greedy_budget_lr1e5_s43_epoch2'),
 ('new_targets_vs_rawhard','sample_augmented_lr1e5_s43_serial_replay_epoch2','sample_augmented_rawhard_lr1e5_s43_epoch2'),
 ('nll_min_vs_random','nll_noguided_min_lr1e5_s43_epoch2','nll_noguided_random_lr1e5_s43_epoch2'),
 ('student_keepgreedy','sample_keepgreedy_lr1e5_s43_epoch2','sample_all_lr1e5_s43_epoch2'),
 ('teacher_keepgreedy','teacher_keepgreedy_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4'),
 ('kl01','teacher_all_kl01_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4'),
 ('kl1','teacher_all_kl1_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4'),
]
out={};seeds=[43,44,45,46,47]
for scoring,vectors in [('original',raw),('strict_last_box',strict)]:
 comparisons=[]
 for label,t,c in pairs:
  d=vectors[t]-vectors[c]
  comparisons.append(dict(label=label,treatment=t,control=c,treatment_correct=int(vectors[t].sum()),control_correct=int(vectors[c].sum()),effect=summarize(d[None,:])))
 families={};matrices={}
 for family,template in [('average1234','avg_teacher7_1234_s{seed}_epoch4'),('epoch4','teacher_all_lr5e5_s{seed}_epoch4')]:
  names=[template.format(seed=seed) for seed in seeds];d=np.stack([vectors[name]-vectors['base'] for name in names]);matrices[family]=d
  families[family]=dict(per_seed_correct=[int(vectors[name].sum()) for name in names],vs_base=summarize(d))
 out[scoring]=dict(base_correct=int(vectors['base'].sum()),comparisons=comparisons,five_seed_families=families,average_minus_epoch4=summarize(matrices['average1234']-matrices['epoch4']))
old=json.loads((S/'audits/teacher_average_five_seed_development.json').read_text())
for family in ['average1234','epoch4']:assert out['original']['five_seed_families'][family]['vs_base']==old['families'][family]['vs_base']
assert out['original']['average_minus_epoch4']==old['average_minus_epoch4']
write(S/'results/post_sampling_primary_analysis.json',dict(time=time.time(),script_sha256=sha(__file__),scoring_audit_sha256=sha(audit_path),results=out,original_five_seed_recomputed_exactly=True,scope='Prespecified fixed-endpoint seed43 developmental contrasts; intervals are unadjusted and do not account for all adaptive exploration. Five-seed families preserve all43..47. Strict last-box sensitivity does not replace the original primary metric. No OOD or full5000 result.'))
print(json.dumps({scoring:{'base':v['base_correct'],'pairs':[{k:r[k] for k in ['label','treatment_correct','control_correct','effect']} for r in v['comparisons']],'five_seed':v['five_seed_families']} for scoring,v in out.items()},indent=2))
