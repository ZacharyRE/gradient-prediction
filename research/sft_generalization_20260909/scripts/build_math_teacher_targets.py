"""Strict math-specialized teacher targets and same-question controls; review required."""
from collections import Counter,defaultdict
import random
from transformers import AutoTokenizer
from common import *
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify

def main():
 root=S/'results/generation/teacher_math7';assert (root/'complete.json').exists();raw=read(S/'data/train.jsonl');tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 assert json.loads((root/'manifest.json').read_text())['input_sha256']==sha(S/'data/train.jsonl')
 generated=read(root/'predictions.jsonl');assert len(generated)==len(raw) and [g['index'] for g in generated]==list(range(len(raw)))
 prior={r['original_index'] for r in json.loads((FOLLOW/'data/self_targets/review_and_final_manifest.json').read_text())['decisions'] if r['decision']=='exclude_pair'};prior.update(json.loads((S/'audits/target_review_decisions.json').read_text())['excluded_original_indices']);p=S/'audits/math_teacher_review_decisions.json';excluded=set(json.loads(p.read_text())['excluded_target_indices']) if p.exists() else set();pool={};reasons=Counter()
 for g in generated:
  i=g['index'];r=raw[i];text=g['prediction'];assert r['problem']==g['problem']
  if i in prior:reasons['prior_pair_exclusion']+=1;continue
  if g['finish_reason']!='stop':reasons['incomplete']+=1;continue
  boxes=boxed_contents(text)
  if not boxes or not verify(parse(r['solution']),parse('\\boxed{'+boxes[-1]+'}')):reasons['incorrect_last_box']+=1;continue
  n=len(tok.encode(text,add_special_tokens=False))+1;prompt=len(chat_prompt_ids(tok,r['problem']))
  if prompt+max(n,len(tok.encode(r['solution'],add_special_tokens=False))+1)>2048:reasons['too_long']+=1;continue
  if any(t in text for t in ['[asy]','\\begin{asy}','```asy','<think>','</think>']):reasons['code_or_thinking']+=1;continue
  pool[i]=dict(r,solution=text,original_index=i,target_source='teacher_math7',target_tokens=n,target_sha256=hashlib.sha256(text.encode()).hexdigest(),candidate_sources=['teacher_math7'])
 automatic=len(pool);pool={i:r for i,r in pool.items() if i not in excluded};sets={'teacher_math7_all':list(pool.values()),'teacher_math7_raw':[dict(raw[i],original_index=i,target_source='raw') for i in pool]}
 known={tuple(x) for x in json.loads((S/'audits/nll_noguided_review_decisions.json').read_text())['excluded_target_keys']}
 others={};known_other_exclusions={}
 for name,file in [('teacher7','teacher_all'),('teacher32','teacher32_all'),('sample','sample_all')]:
  rr=read(S/f'data/{file}.jsonl');bad=[r['original_index'] for r in rr if (r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest()) in known]
  others[name]={r['original_index']:r for r in rr if r['original_index'] not in bad};known_other_exclusions[name]=bad
 for name,other in others.items():
  ix=sorted(set(pool)&set(other));sets['teacher_math7_common_'+name]=[pool[i] for i in ix];sets[name+'_common_math7']=[other[i] for i in ix]
 audit={}
 for name,rr in sets.items():
  jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in rr))
 base=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl');groups=defaultdict(list)
 for i,r in pool.items():groups['greedy_right' if base[i]['correct'] else 'greedy_wrong_general7_covered' if i in others['teacher7'] else 'greedy_wrong_general7_uncovered'].append(r)
 review=S/'audits/math_teacher_review.jsonl'
 if not review.exists():
  rng=random.Random(20261008);samples=[]
  for group,rr in sorted(groups.items()):
   for r in rng.sample(rr,min(8,len(rr))):samples.append(dict(r,review_group=group,raw_solution=raw[r['original_index']]['solution']))
  jsonl(review,samples)
 write(S/'audits/math_teacher_build.json',dict(time=time.time(),automatic=automatic,review_excluded=sorted(excluded),known_other_target_exclusions=known_other_exclusions,accepted=len(pool),historical_greedy_wrong=sum(not base[i]['correct'] for i in pool),rejections=dict(reasons),datasets=audit,review_groups={g:len(v) for g,v in groups.items()},review_strata_note='General7 coverage after excluding already-proven bad selected targets; original immutable general7 full training set unchanged.',status='Provisional; explicit process review release before any training.'))
 print(json.dumps(dict(automatic=automatic,accepted=len(pool),hard=sum(not base[i]['correct'] for i in pool),datasets={n:len(r) for n,r in sets.items()}),indent=2))
if __name__=='__main__':main()
