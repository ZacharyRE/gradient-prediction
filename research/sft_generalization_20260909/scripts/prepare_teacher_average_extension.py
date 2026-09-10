"""Prospectively extend the fixed promising averaged-teacher recipe to seeds46/47."""
from common import *

assert not (S/'results/confirmation_plan.json').exists()
assert not list((S/'results/evaluation').glob('ood_*/*/math_summary.json'))
results=json.loads((S/'audits/fixed_seed_math_reused1500.json').read_text())
family=results['families']['teacher7_avg1234'];assert not family['missing'] and family['all_seeds_positive']
qp=S/'results/train_queue.json';queue=json.loads(qp.read_text());control=next(j for j in queue if j['name']=='teacher_all_lr5e5_s43')
jobs=[dict(control,name=f'teacher_all_lr5e5_s{seed}',seed=seed) for seed in [46,47]]
assert not set(j['name'] for j in jobs)&set(j['name'] for j in queue)
position=next(i for i,j in enumerate(queue) if j['name']=='nll_noguided_min_lr5e5_s43');queue[position:position]=jobs
ap=S/'audits/epoch_average_plan.json';averaging=json.loads(ap.read_text());new=[]
for seed in [46,47]:
 job=dict(name=f'avg_teacher7_1234_s{seed}',endpoint=4,sources=[str(S/f'results/training/teacher_all_lr5e5_s{seed}/epoch{epoch}') for epoch in range(1,5)])
 assert job['name'] not in [j['name'] for j in averaging['jobs']];new.append(job)
write(S/'audits/teacher_average_seed_extension_plan.json',dict(time=time.time(),training_jobs=jobs,derived_jobs=new,data_sha256=sha(control['train_file']),primary_extension_endpoint='Unweighted mean of effectiveLoRAupdates at epochs1,2,3,4; exactconcatenation implementation unchanged. Epoch4unaveraged is a fixed diagnostic control.',fixed_seed_list=[43,44,45,46,47],seen_three_seed_reused1500=family,development_result_sha256=sha(S/'audits/fixed_seed_math_reused1500.json'),original_average_plan_sha256=sha(ap),queue_total=len(queue),scope='Prospective extra-seed development extension of one promising candidate; final primaryrecipe not selected. Originalthree seeds and everyfutureseed remain in reports. NofreshOOD/full5000results.',limitation='Unadjusted three-seed reused1500 meanCIpositive; four-family seed-questionBonferroni interval still crosseszero. This is a reason to replicate, not general efficacy evidence.'))
averaging['jobs'].extend(new);write(ap,averaging);write(qp,queue)
pp=S/'results/dev_priority.json';priority=json.loads(pp.read_text());added=[f'avg_teacher7_1234_s{seed}_epoch4' for seed in [46,47]]+[f'teacher_all_lr5e5_s{seed}_epoch4' for seed in [46,47]]
# Preserve initial math/retention-control priorities, then examine the predeclared extension.
position=next(i for i,n in enumerate(priority) if n=='nll_noguided_min_lr1e5_s43_epoch1');priority[position:position]=added;assert len(priority)==len(set(priority));write(pp,priority)
print(json.dumps(dict(queue_total=len(queue),jobs=jobs,averages=new)))
