"""Build new strong-teacher datasets without changing earlier released datasets."""
import random
from collections import Counter
from common import *
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify
def main():
 root=S/'results/generation/teacher32';assert (root/'complete.json').exists()
 raw=read(S/'data/train.jsonl');base=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl')
 previous=read(S/'data/previous_self.jsonl');previous_q={r['problem']:r for r in previous}
 older_excluded={r['original_index'] for r in json.loads((FOLLOW/'data/self_targets/review_and_final_manifest.json').read_text())['decisions'] if r['decision']=='exclude_pair'}
 older_excluded.update(json.loads((S/'audits/target_review_decisions.json').read_text())['excluded_original_indices'])
 decisions=S/'audits/teacher32_review_decisions.json';new_excluded=set(json.loads(decisions.read_text())['excluded_target_indices']) if decisions.exists() else set()
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);pool={};reasons=Counter()
 for g in read(root/'predictions.jsonl'):
  i=g['index'];r=raw[i];assert r['problem']==g['problem'];text=g['prediction'];boxes=boxed_contents(text)
  if i in older_excluded:reasons['prior_pair_excluded']+=1;continue
  if g['finish_reason']!='stop':reasons['incomplete']+=1;continue
  if not boxes or not verify(parse(r['solution']),parse('\\boxed{'+boxes[-1]+'}')):reasons['incorrect_last_box']+=1;continue
  prompt=len(chat_prompt_ids(tok,r['problem']));n=len(tok.encode(text,add_special_tokens=False))+1
  if n+prompt>2048 or len(tok.encode(r['solution'],add_special_tokens=False))+1+prompt>2048:reasons['too_long']+=1;continue
  if '[asy]' in text or '\\begin{asy}' in text:reasons['diagram_target']+=1;continue
  if '<think>' in text or '</think>' in text:reasons['unexpected_thinking_tags']+=1;continue
  pool[i]=dict(r,solution=text,original_index=i,generation_sample=0,target_source='teacher32',target_tokens=n)
 automatic_n=len(pool)
 pool={i:r for i,r in pool.items() if i not in new_excluded}
 sample={r['original_index']:r for r in read(S/'data/sample_all.jsonl')};teacher={r['original_index']:r for r in read(S/'data/teacher_all.jsonl')}
 sets={'teacher32_all':list(pool.values()),'teacher32_raw':[dict(raw[i],original_index=i,target_source='raw') for i in pool]}
 for other,label in [(sample,'sample'),(teacher,'teacher7')]:
  ids=sorted(set(pool)&set(other));sets['teacher32_common_'+label]=[pool[i] for i in ids];sets[label+'_common_teacher32']=[other[i] for i in ids]
 added=[i for i in pool if raw[i]['problem'] not in previous_q]
 sets['teacher32_augmented']=previous+[pool[i] for i in added]
 sets['teacher32_augmented_rawhard']=previous+[dict(raw[i],original_index=i,target_source='raw') for i in added]
 sets['teacher32_greedy_budget']=previous+[dict(previous[j%len(previous)]) for j in range(len(added))]
 audit={}
 for name,rr in sets.items():
  jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in rr),levels=dict(Counter(r['level'] for r in rr)))
 review_path=S/'audits/teacher32_review.jsonl'
 groups={'greedy_right':[r for i,r in pool.items() if base[i]['correct']], 'greedy_wrong_teacher7_covered':[r for i,r in pool.items() if not base[i]['correct'] and i in teacher], 'greedy_wrong_teacher7_uncovered':[r for i,r in pool.items() if not base[i]['correct'] and i not in teacher]}
 if not review_path.exists():
  rng=random.Random(20260917);review=[]
  for name,rr in groups.items():
   for r in rng.sample(rr,min(8,len(rr))):review.append(dict(r,review_group=name,raw_solution=raw[r['original_index']]['solution']))
  jsonl(review_path,review)
 write(S/'audits/teacher32_build.json',dict(time=time.time(),automatic_accepted=automatic_n,new_review_excluded=sorted(new_excluded),rejections=dict(reasons),final_n=len(pool),historical_greedy_wrong=sum(not base[i]['correct'] for i in pool),review_group_sizes={k:len(v) for k,v in groups.items()},datasets=audit,status='Requires recorded manual review release before training'))
 print(json.dumps(dict(automatic_accepted=automatic_n,final_n=len(pool),hard=sum(not base[i]['correct'] for i in pool),review_group_sizes={k:len(v) for k,v in groups.items()},datasets={k:v['n'] for k,v in audit.items()}),indent=2))
if __name__=='__main__':main()
