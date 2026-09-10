"""Same-question retention target control, using only already released targets."""
from transformers import AutoTokenizer
from common import *
from training.train_lora_sft import chat_prompt_ids
raw=read(S/'data/train.jsonl');byq={r['problem']:i for i,r in enumerate(raw)}
known={tuple(x) for x in json.loads((S/'audits/nll_noguided_review_decisions.json').read_text())['excluded_target_keys']}
greedy={};excluded=[]
for r in read(S/'data/previous_self.jsonl'):
 i=byq[r['problem']];key=(i,hashlib.sha256(r['solution'].encode()).hexdigest())
 if key in known:excluded.append(i)
 else:greedy[i]=r
base=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl')
assert all(base[i]['correct'] for i in greedy)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);jobs=[];datasets={}
for source,lr,label,epoch in [('sample',1e-5,'1e5',2),('teacher',5e-5,'5e5',4)]:
 original=read(S/f'data/{source}_all.jsonl');rows=[];replaced=[];changed=[]
 for r in original:
  i=r['original_index'];rr=dict(r)
  if i in greedy:
   rr.update(solution=greedy[i]['solution'],target_source='retained_prior_greedy',target_sha256=hashlib.sha256(greedy[i]['solution'].encode()).hexdigest(),candidate_sources=['greedy'])
   replaced.append(i)
   if rr['solution']!=r['solution']:changed.append(i)
  else:rr['target_source']=source+'_unchanged_retention_control'
  rr['target_tokens']=len(tok.encode(rr['solution'],add_special_tokens=False))+1
  assert len(chat_prompt_ids(tok,rr['problem']))+rr['target_tokens']<=2048
  rows.append(rr)
 assert [r['problem'] for r in rows]==[r['problem'] for r in original]
 assert all(a['solution']==b['solution'] for a,b in zip(rows,original) if not base[a['original_index']]['correct'])
 name=source+'_keepgreedy';path=S/f'data/{name}.jsonl';jsonl(path,rows)
 datasets[name]=dict(n=len(rows),sha256=sha(path),control_sha256=sha(S/f'data/{source}_all.jsonl'),greedy_available=len(replaced),text_changed=len(changed),changed_indices=changed,hard_question_targets_unchanged=True,hard_questions=sum(not base[r['original_index']]['correct'] for r in rows),target_tokens=sum(r['target_tokens'] for r in rows),control_target_tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in original),primary_matched_epoch=epoch)
 jobs.append(dict(name=f'{name}_lr{label}_s43',train_file=str(path),lr=lr,stop=4))
qpath=S/'results/train_queue.json';queue=json.loads(qpath.read_text());assert not any(j['name'] in {q['name'] for q in queue} for j in jobs)
idx=next(i for i,j in enumerate(queue) if j['name']=='teacher_all_kl1_lr5e5_s43')+1;queue[idx:idx]=jobs
write(S/'audits/keep_greedy_plan.json',dict(time=time.time(),datasets=datasets,jobs=jobs,queue_total=len(queue),known_bad_prior_greedy_excluded=excluded,prior_greedy_sha256=sha(S/'data/previous_self.jsonl'),hypothesis='Replacing alternative solutions on already greedy-correct questions may disturb retention; preserve released original greedy targets while keeping wrong-question supervision exactly unchanged.',design='Exact source-control question set/order, initialization seed, optimizer and exposure; target length and content on changed already-correct questions differ.',distinction='Unlike earlier1576-row augmentation, these controls match original1563/1547-row source datasets exactly. No new generations or OOD outcomes.',interpretation='Retained-target pipeline intervention, not isolated KL/probability/length effect. Pilots only; multiple seeds and frozen confirmation required for efficacy.'))
write(qpath,queue)
print(json.dumps(dict(queue_total=len(queue),datasets=datasets),indent=2))
