"""Freeze bounded final selection/controls before four new seed outcomes exist."""
from common import *
p=S/'audits/final_selection_rule.json';assert not p.exists() and not (S/'results/confirmation_plan.json').exists()
seeds=[43,44,45,46,47]
families={
 'teacher7_avg1234':{str(seed):f'avg_teacher7_1234_s{seed}_epoch4' for seed in seeds},
 'teacher7_expanded_avg1234':{str(seed):f'avg_teacher7_expansion_combined_1234_s{seed}_epoch4' for seed in seeds},
 'teacher7_epoch4':{str(seed):f'teacher_all_lr5e5_s{seed}_epoch4' for seed in seeds}}
for seed in seeds[1:]:
 assert not (S/f'results/training/teacher7_expansion_combined_lr5e5_s{seed}').exists()
 assert not (S/'results/evaluation/dev'/families['teacher7_expanded_avg1234'][str(seed)]).exists()
for d in ['math_reused5000','ood_minerva','ood_olympiad','ood_svamp','ood_amc23']:
 assert not list((S/'results/evaluation'/d).glob('*/math_predictions.jsonl'))
write(p,dict(time=time.time(),script_sha256=sha(__file__),seeds=seeds,families=families,default_primary='teacher7_avg1234',alternative_primary='teacher7_expanded_avg1234',rule='Choose expanded ONLY IF every one of its five fixed seeds strictly exceeds baseline on development500 under BOTH original and strict scoring, AND expanded mean correctness strictly exceeds old-average mean under BOTH scoring rules. Otherwise keep old-average primary. No p-value selection, seed dropping, endpoint change, or futureMATH5000/OOD inspection.',controls='All three fixed families evaluated on final benchmarks regardless of selection. Old unaveragedepoch4 isolates average-versus-endpoint for old data; old-versus-expanded averages compare the whole expansion recipe, not purecoverage atfixedtokens. Only selected primary has the prespecified efficacy claim; secondary intervals remain exploratory/unadjusted.',required_before_selection=['matched_generator_control_analysis.json','expanded_seed_replication_dispatch.json with all_complete','all16 dev summaries and exact-protocol strict scoring','final_statistics_contract.json passed'],replication_plan_sha256=sha(S/'audits/teacher7_expansion_seed_replication_plan.json'),known_evidence_sha256={f:sha(S/f) for f in ['audits/teacher_average_five_seed_math_reused1500.json','audits/coverage_math1500_analysis.json','audits/teacher_average_five_seed_development.json']},scope='Prospective deterministic choice using development data. MATH5000 remains historically reused; only reservedOOD is fresh. Five-seed positivity and both question/seed-question95%CIs on MATH and mathematicalOODmacro stay unchanged. Maximum16 models includingbaseline; native5primary+base.'))
print(p)
