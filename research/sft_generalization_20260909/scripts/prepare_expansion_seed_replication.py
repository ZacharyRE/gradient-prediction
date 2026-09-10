"""Predeclare four additional seeds for the existing expanded teacher candidate."""
from common import *
assert not (S/'results/confirmation_plan.json').exists()
for d in ['math_reused5000','ood_minerva','ood_olympiad','ood_svamp','ood_amc23']:
 assert not list((S/'results/evaluation'/d).glob('*/math_predictions.jsonl'))
path=S/'audits/teacher7_expansion_seed_replication_plan.json';assert not path.exists()
queue_path=S/'results/train_queue.json';queue=json.loads(queue_path.read_text())
source=next(j for j in queue if j['name']=='teacher7_expansion_combined_lr5e5_s43')
jobs=[dict(source,name=f'teacher7_expansion_combined_lr5e5_s{seed}',seed=seed) for seed in [44,45,46,47]]
assert not {j['name'] for j in jobs}&{j['name'] for j in queue}
ap=S/'audits/epoch_average_plan.json';average=json.loads(ap.read_text());derived=[]
for seed in [44,45,46,47]:
 derived.append(dict(name=f'avg_teacher7_expansion_combined_1234_s{seed}',endpoint=4,sources=[str(S/f'results/training/teacher7_expansion_combined_lr5e5_s{seed}/epoch{e}') for e in range(1,5)]))
plan=dict(time=time.time(),script_sha256=sha(__file__),jobs=jobs,derived_jobs=derived,seeds=[43,44,45,46,47],data_sha256=sha(source['train_file']),trainer_sha256=sha(S/'scripts/train.py'),averager_sha256=sha(S/'scripts/average_epoch_adapters.py'),queue_before_sha256=sha(queue_path),average_plan_before_sha256=sha(ap),wait_for_training='teacher_common_lr5e5_s43',prior_evidence_sha256={f:sha(S/f) for f in ['audits/coverage_math1500_analysis.json','audits/coverage_math1500_sensitivity.json','results/coverage_dev_fixed_analysis.json','audits/teacher_average_five_seed_math_reused1500.json']},reason='Expanded average seed43 is309/500 and852/1500; old average43 is301/500 and845/1500. Neither expanded-minus-old nor coverage-minus-repeat averaged contrast establishes a benefit. Four fixed additional seeds test stability of this bounded candidate; no new recipe/hyperparameter search. Existing old five-seed candidate remains eligible for final selection before any final outputs.',endpoint='Unweighted effective-update mean of epochs1,2,3,4; every seed retained. Fixedepoch4 saved as diagnostic.',handoff='Wait for six matched-generator controls, STOP_TRAINER and natural exit of old training queue, then two quiet GPU1 checks. Keep STOP_TRAINER set and run only these four explicit jobs. No low-priority queue resume.',limits='Only GPU1 for training; current GPU3 evaluator retains ownership. Start before06:00UTC, finish before07:00UTC or fail closed; reserve at least8h for final evidence/report. No new1500 extension is scheduled for these four replicas.')
write(path,plan)
backup=S/'audits/epoch_average_plan_before_expansion_seed_replication.json';assert not backup.exists();backup.write_bytes(ap.read_bytes())
average['jobs'].extend(derived);write(ap,average)
# Append for complete experiment accounting. STOP_TRAINER keeps the old queue paused.
write(queue_path,queue+jobs)
pp=S/'results/dev_priority.json';priority=json.loads(pp.read_text());new=[j['name']+'_epoch4' for j in derived]+[j['name']+'_epoch4' for j in jobs]
assert not set(new)&set(priority)
last=priority.index('teacher_common_lr5e5_s43_epoch4')+1;priority[last:last]=new;write(pp,priority)
print(json.dumps(dict(plan=str(path),four_training_jobs=[j['name'] for j in jobs],queue_total=len(queue)+len(jobs))))
