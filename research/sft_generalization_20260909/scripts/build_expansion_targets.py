"""Strict filtered new-question expansion and exposure-count controls; no eval labels."""
import random
from collections import Counter,defaultdict
from common import *
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify

def main():
 gen=S/'results/generation/teacher32_expansion';assert (gen/'complete.json').exists()
 src=S/'data/expansion_ready.jsonl';raw=read(src);assert json.loads((gen/'manifest.json').read_text())['input_sha256']==sha(src)
 original_all=read(S/'data/teacher32_all.jsonl')
 known={tuple(x) for x in json.loads((S/'audits/nll_noguided_review_decisions.json').read_text())['excluded_target_keys']}
 core_excluded=[r['original_index'] for r in original_all if (r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest()) in known]
 original=[r for r in original_all if r['original_index'] not in core_excluded]
 oldproblems={r['problem'] for r in read(S/'data/train.jsonl')};assert not oldproblems&{r['problem'] for r in raw}
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);pool={};reasons=Counter();decision=S/'audits/expansion_target_review_decisions.json';excluded=set(json.loads(decision.read_text())['excluded_expansion_indices']) if decision.exists() else set()
 for g in read(gen/'predictions.jsonl'):
  i=g['index'];r=raw[i];assert r['problem']==g['problem'];text=g['prediction'];boxes=boxed_contents(text)
  if g['finish_reason']!='stop':reasons['incomplete']+=1;continue
  if not boxes or not verify(parse(r['solution']),parse('\\boxed{'+boxes[-1]+'}')):reasons['incorrect_last_box']+=1;continue
  prompt=len(chat_prompt_ids(tok,r['problem']));n=len(tok.encode(text,add_special_tokens=False))+1
  if n+prompt>2048 or len(tok.encode(r['solution'],add_special_tokens=False))+1+prompt>2048:reasons['too_long']+=1;continue
  if any(x in text for x in ['[asy]','\\begin{asy}','```asy']):reasons['diagram_code_target']+=1;continue
  if '<think>' in text or '</think>' in text:reasons['thinking_tags']+=1;continue
  # Expansion indices are local to expansion_ready, NEVER original2000 indices.
  rr=dict(r,solution=text,expansion_index=i,target_source='teacher32_expansion',target_tokens=n)
  rr.pop('original_index',None);pool[i]=rr
 automatic=len(pool);pool={i:r for i,r in pool.items() if i not in excluded};new=list(pool.values());combined=original+new
 rawnew=[dict(raw[i],expansion_index=i,target_source='raw_expansion',target_tokens=len(tok.encode(raw[i]['solution'],add_special_tokens=False))+1) for i in pool]
 repeat=[dict(original[i%len(original)],repeat_slot=i,target_source='teacher32_exposure_control') for i in range(len(combined))]
 sets=dict(expansion32_new=new,expansion32_combined=combined,expansion32_repeat_control=repeat,expansion32_rawnew_control=original+rawnew)
 assert len(combined)==len(repeat)==len(sets['expansion32_rawnew_control']);assert [r['problem'] for r in combined]==[r['problem'] for r in sets['expansion32_rawnew_control']]
 audit={}
 for name,rr in sets.items():
  jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(r['target_tokens'] for r in rr),levels=dict(Counter(r['level'] for r in rr)))
 groups=defaultdict(list)
 for r in new:
  group='diagram' if '[asy]' in r['problem'] else 'level5' if r['level']=='Level 5' else 'level34' if r['level'] in ['Level 3','Level 4'] else 'level12';groups[group].append(r)
 review=S/'audits/expansion_target_review.jsonl'
 if not review.exists():
  rng=random.Random(20260925);reviews=[]
  for group,rr in sorted(groups.items()):
   for r in rng.sample(rr,min(8,len(rr))):reviews.append(dict(r,review_group=group,raw_solution=raw[r['expansion_index']]['solution']))
  jsonl(review,reviews)
 write(S/'audits/expansion_target_build.json',dict(time=time.time(),input_sha256=sha(src),generation_sha256=sha(gen/'predictions.jsonl'),automatic_accepted=automatic,old_core_known_error_exclusions=core_excluded,old_core_n=len(original),review_excluded=sorted(excluded),final_new=len(new),rejections=dict(reasons),datasets=audit,review_groups={k:len(v) for k,v in groups.items()},status='Provisional pending review and explicit release',index_contract='expansion_index is local to expansion_ready; source_row_index records original HF train row. No joins to original2000 by expansion_index.',control_limitations='Repeat matches example exposures, batch boundaries, horizon and base targets, but not total target tokens or new-question distribution. Rawnew control matches exact questions/order but differs target length and source.'))
 print(json.dumps(dict(automatic=automatic,new=len(new),sizes={k:len(v) for k,v in sets.items()},rejections=dict(reasons)),indent=2))
if __name__=='__main__':main()
