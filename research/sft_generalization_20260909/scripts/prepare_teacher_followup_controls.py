"""Bounded controls for the promising general7B target recipe, before their scores."""
from common import *
plan=S/'audits/teacher_followup_controls_plan.json';assert not plan.exists()
queuepath=S/'results/train_queue.json';queue=json.loads(queuepath.read_text())
jobs=[]
for lr,label in [(1e-5,'1e5'),(5e-5,'5e5')]:
 jobs.append(dict(name=f'teacher_all_qkvo64_lr{label}_s43',train_file=str(S/'data/teacher_all.jsonl'),lr=lr,rank=64,alpha=64,modules='qkvo',stop=4))
for coefficient,label in [(0.,'0'),(1.,'1')]:
 jobs.append(dict(name=f'teacher_all_dftkl{label}_lr5e5_s43',train_file=str(S/'data/teacher_all.jsonl'),lr=5e-5,objective='dft',kl_coefficient=coefficient,trainer_script='train_anchored.py',stop=4))
assert not any(j['name'] in {q['name'] for q in queue} for j in jobs)
for j in jobs:assert not (S/'results/training'/j['name']).exists()
idx=next(i for i,j in enumerate(queue) if j['name']=='expansion32_combined_lr1e5_s43');queue[idx:idx]=jobs
ap=S/'audits/epoch_average_plan.json';original=ap.read_bytes();avg=json.loads(original)
backup=S/'audits/epoch_average_plan_before_expansion_extension.json';assert not backup.exists();backup.write_bytes(original)
derived=[]
for source in ['teacher7_expansion_combined','teacher7_expansion_repeat_control','teacher7_expansion_rawnew_control']:
 run=source+'_lr5e5_s43'
 assert not (S/'results/training'/run).exists()
 derived.append(dict(name='avg_'+source+'_1234_s43',endpoint=4,sources=[str(S/'results/training'/run/f'epoch{e}') for e in range(1,5)]))
assert not any(j['name'] in {x['name'] for x in avg['jobs']} for j in derived)
avg['jobs']+=derived
prioritypath=S/'results/dev_priority.json';priority=json.loads(prioritypath.read_text());idx=priority.index('expansion32_combined_lr1e5_s43_epoch4')
priority[idx:idx]=[j['name']+'_epoch4' for j in derived+jobs]
write(plan,dict(time=time.time(),jobs=jobs,teacher_data_sha256=sha(S/'data/teacher_all.jsonl'),trainer_sha256=sha(S/'scripts/train_anchored.py'),original_train_sha256=sha(S/'scripts/train.py'),derived_jobs=derived,prior_average_plan_sha256=hashlib.sha256(original).hexdigest(),averager_sha256=sha(S/'scripts/average_epoch_adapters.py'),rationale=dict(rank='Existing high-rank new-target jobs use student/32B, while the promising general7B recipe and its exact four-epoch average use QKVO. Compare learned rank64 QKVO against same-source rank16 at both already-used LRs; alpha/r=1, identical zero initial function but A shapes differ. This does not isolate averaging from every optimization change.',objective='Earlier CE+KL is not full ASFT. Add paired DFT and DFT+forwardKL1 on exactly the existing teacher targets and optimizer, fixed LR5e-5/epoch4. Uses already analytically checked DFT+KL helper; not claimed as paper-scale reproduction.',scale_average='Apply the already defined exact DeltaW mean of epochs1..4 to all three not-yet-started7B scale arms, before any scale training scores, to avoid choosing favorable epochs.'),scope='Four additional single-seed development training controls and three CPU-derived averages. No fresh OOD/5000 outputs and no primary-family selection. Fixed endpoints, all negative results preserved. Exploration cutoff still06:30UTC.'))
write(ap,avg);write(prioritypath,priority);write(queuepath,queue)
print(json.dumps(dict(queue=len(queue),new_training=4,new_derived=3)))
