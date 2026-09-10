"""Matched reference-target controls for the newly evaluated three-seed families."""
from common import *
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids

tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
qp=S/'results/train_queue.json';queue=json.loads(qp.read_text());jobs=[];datasets={}
for kind,lr,epoch in [('sample','1e5',2),('teacher','5e5',4)]:
 source=S/f'data/{kind}_all.jsonl';reference=S/f'data/{kind}_raw.jsonl';a=read(source);b=read(reference)
 assert [(r['original_index'],r['problem']) for r in a]==[(r['original_index'],r['problem']) for r in b]
 raw=read(S/'data/train.jsonl');assert all(r['solution']==raw[r['original_index']]['solution'] for r in b)
 lengths=[len(chat_prompt_ids(tok,r['problem']))+len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in b];assert max(lengths)<=2048
 control=next(j for j in queue if j['name']==f'{kind}_all_lr{lr}_s43')
 job=dict(control,name=f'{kind}_matchedraw_lr{lr}_s43',train_file=str(reference));jobs.append(job)
 datasets[kind]=dict(n=len(a),source=str(source),source_sha256=sha(source),reference=str(reference),reference_sha256=sha(reference),source_tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in a),reference_tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in b),max_total_reference_tokens=max(lengths),primary_epoch=epoch)
assert not set(j['name'] for j in jobs)&set(j['name'] for j in queue)
# Retention and within-question NLL comparisons keep their previously chosen priority.
position=next(i for i,j in enumerate(queue) if j['name']=='sample_all_shortcos2_lr1e5_s43')
queue[position:position]=jobs
write(S/'audits/raw_source_controls_plan.json',dict(time=time.time(),jobs=jobs,datasets=datasets,queue_total=len(queue),motivation='New families differ from historical1175-question self targets in coverage and solution source. These controls isolate target-pipeline replacement on exactly each new family question set, conditional on fixed LR/seed/exposure.',known_outcomes_at_release='Sample fixed three-seed1500 scores826/844/831 versusbase818; teacher7seed43=859, otherteacher seeds pending. All development outcomes already available are exploratory.',limits=['Different target content and token length; not isolated prose-style effect.','Original reference solutions and immutable source targets may contain known process errors.','Two single-seed pilots only, not independently established causal generalization.','Historical1175 raw/self intervention is reused evidence, not rerun here; these1563/1547 sets include newly covered baseline-wrong questions.']))
write(qp,queue)
pp=S/'results/dev_priority.json';priority=json.loads(pp.read_text());added=[f'sample_matchedraw_lr1e5_s43_epoch2',f'teacher_matchedraw_lr5e5_s43_epoch4'];write(pp,priority+[n for n in added if n not in priority])
print(json.dumps(dict(queue_total=len(queue),datasets=datasets,jobs=jobs)))
