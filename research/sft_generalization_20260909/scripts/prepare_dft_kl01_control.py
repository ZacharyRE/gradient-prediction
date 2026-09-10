"""Complete the already-used CE/DFT x KL0/0.1/1 comparison before DFT scores."""
from common import *
planpath=S/'audits/dft_kl01_control_plan.json'
assert not planpath.exists() and not (S/'results/confirmation_plan.json').exists()
for label in ['0','1']:
    assert not (S/'results/training'/f'teacher_all_dftkl{label}_lr5e5_s43').exists()
name='teacher_all_dftkl01_lr5e5_s43'
job=dict(name=name,train_file=str(S/'data/teacher_all.jsonl'),lr=5e-5,
         objective='dft',kl_coefficient=.1,trainer_script='train_anchored.py',stop=4)
queuepath=S/'results/train_queue.json';queue=json.loads(queuepath.read_text())
assert name not in {j['name'] for j in queue}
index=next(i for i,j in enumerate(queue) if j['name']=='teacher_all_dftkl1_lr5e5_s43')+1
queue.insert(index,job)
pp=S/'results/dev_priority.json';priority=json.loads(pp.read_text())
index=priority.index('teacher_all_dftkl1_lr5e5_s43_epoch4')+1
priority.insert(index,name+'_epoch4')
write(planpath,dict(time=time.time(),job=job,data_sha256=sha(S/'data/teacher_all.jsonl'),
    trainer_sha256=sha(S/'scripts/train_anchored.py'),fixed_endpoint=4,seed=43,
    preceding_queue_sha256=sha(queuepath),preceding_priority_sha256=sha(pp),
    related_plan_sha256=sha(S/'audits/teacher_followup_controls_plan.json'),
    source='https://arxiv.org/html/2509.23753v3',
    rationale='ASFT reports a coefficient optimum around0.1 in medical QA, not a demonstrated optimum for this local mathematics task. CE already has0/0.1/1; add DFT0.1 before any DFT training or scores to complete the matched objective-by-coefficient controls. Same1547 targets, LR5e-5, rank16QKVO, horizon8 and fixedepoch4. No new data or inference metric.',
    scope='One additional single-seed development pilot; no primary selection, independent seed claim or paper-scale reproduction.'))
write(pp,priority);write(queuepath,queue)
print(json.dumps(dict(queue=len(queue),added=name)))
