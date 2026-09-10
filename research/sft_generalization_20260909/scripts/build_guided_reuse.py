"""Reuse accepted historical reference-guided targets; separate raw fallbacks."""
import random
from collections import Counter
from common import *
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify
LEGACY=ROOT/'result/MATH/Qwen2.5-1.5B-Instruct/sft_pilot/precision_style_ablation_20260907/rewrites'
def main():
 raw=read(S/'data/train.jsonl');old=read(LEGACY/'train_rewritten.jsonl');generations=read(LEGACY/'raw_rewrites.jsonl');assert len(raw)==len(old)==len(generations)==2000
 assert [r['problem'] for r in raw]==[r['problem'] for r in old]
 base=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl');previous=read(S/'data/previous_self.jsonl');previous_q={r['problem']:r for r in previous}
 sample={r['original_index']:r for r in read(S/'data/sample_all.jsonl')};teacher={r['original_index']:r for r in read(S/'data/teacher_all.jsonl')}
 excluded={r['original_index'] for r in json.loads((FOLLOW/'data/self_targets/review_and_final_manifest.json').read_text())['decisions'] if r['decision']=='exclude_pair'};excluded.update(json.loads((S/'audits/target_review_decisions.json').read_text())['excluded_original_indices'])
 review_decisions=S/'audits/guided_review_decisions.json';extra=set(json.loads(review_decisions.read_text())['excluded_target_indices']) if review_decisions.exists() else set()
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);pool={};reasons=Counter()
 for i,(r,g) in enumerate(zip(old,generations)):
  assert g['index']==i
  if not r['style_rewrite_accepted']:reasons['historical_fallback_not_guided_target']+=1;continue
  assert r['solution']==g['prediction']
  if i in excluded:reasons['prior_pair_exclusion']+=1;continue
  text=r['solution'];boxes=boxed_contents(text)
  if g['finish_reason']!='stop' or not boxes or not verify(parse(raw[i]['solution']),parse('\\boxed{'+boxes[-1]+'}')):reasons['strict_final_verification']+=1;continue
  prompt=len(chat_prompt_ids(tok,r['problem']));n=len(tok.encode(text,add_special_tokens=False))+1
  if prompt+n>2048 or prompt+len(tok.encode(raw[i]['solution'],add_special_tokens=False))+1>2048:reasons['paired_raw_length']+=1;continue
  if '[asy]' in text or '\\begin{asy}' in text:reasons['diagram_target']+=1;continue
  pool[i]=dict(raw[i],solution=text,original_index=i,target_source='guided_legacy',target_tokens=n)
 automatic=len(pool);pool={i:r for i,r in pool.items() if i not in extra}
 datasets={'guided_all':list(pool.values()),'guided_raw':[dict(raw[i],original_index=i,target_source='raw') for i in pool]}
 added=[i for i in pool if raw[i]['problem'] not in previous_q]
 datasets['guided_augmented']=previous+[pool[i] for i in added]
 datasets['guided_augmented_rawhard']=previous+[dict(raw[i],original_index=i,target_source='raw') for i in added]
 datasets['guided_greedy_budget']=previous+[dict(previous[j%len(previous)]) for j in range(len(added))]
 hybrid=read(S/'data/sample_augmented.jsonl');covered={r['problem'] for r in hybrid};new=[i for i in pool if raw[i]['problem'] not in covered]
 datasets['sample_guided_augmented']=hybrid+[pool[i] for i in new]
 datasets['sample_guided_rawhard']=hybrid+[dict(raw[i],original_index=i,target_source='raw') for i in new]
 datasets['sample_guided_budget']=hybrid+[dict(hybrid[j%len(hybrid)]) for j in range(len(new))]
 for other,label in [(sample,'sample'),(teacher,'teacher7')]:
  ids=sorted(set(pool)&set(other));datasets['guided_common_'+label]=[pool[i] for i in ids];datasets[label+'_common_guided']=[other[i] for i in ids]
 q_to_i={r['problem']:i for i,r in enumerate(raw)};audit={}
 for name,rr in datasets.items():
  jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in rr),historical_greedy_wrong=sum(not base[q_to_i[r['problem']]]['correct'] for r in rr),levels=dict(Counter(r['level'] for r in rr)))
 groups={'greedy_right':[r for i,r in pool.items() if base[i]['correct']], 'greedy_wrong_sample_or_teacher7_covered':[r for i,r in pool.items() if not base[i]['correct'] and i in set(sample)|set(teacher)], 'greedy_wrong_neither_covered':[r for i,r in pool.items() if not base[i]['correct'] and i not in set(sample)|set(teacher)]}
 path=S/'audits/guided_review.jsonl'
 if not path.exists():
  rng=random.Random(20260918);review=[]
  for name,rr in groups.items():
   for r in rng.sample(rr,min(8,len(rr))):review.append(dict(r,review_group=name,raw_solution=raw[r['original_index']]['solution']))
  jsonl(path,review)
 write(S/'audits/guided_build.json',dict(time=time.time(),legacy_source=str(LEGACY),legacy_sha256={name:sha(LEGACY/name) for name in ['train_rewritten.jsonl','raw_rewrites.jsonl','protocol.json','summary.json']},automatic_accepted=automatic,excluded_target_indices=sorted(extra),rejections=dict(reasons),datasets=audit,review_group_sizes={k:len(v) for k,v in groups.items()},status='Provisional until recorded assistant process review; existing data generation reused, no new GPU generation'))
 print(json.dumps({k:{x:v[x] for x in ['n','tokens','historical_greedy_wrong']} for k,v in audit.items()},indent=2))
if __name__=='__main__':main()
