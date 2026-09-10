"""Larger correct student-sampled targets from unused officialtrain questions; paired scale/source controls."""
from collections import Counter,defaultdict
import random
from transformers import AutoTokenizer
from common import *
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify

def main():
 gen=S/'results/generation/sample_expansion';assert (gen/'complete.json').exists();basefile=S/'results/evaluation/expansion_ready/base/math_predictions.jsonl';assert (basefile.parent/'math_summary.json').exists()
 source=S/'data/expansion_ready.jsonl';raw=read(source);base=read(basefile);assert len(raw)==len(base)
 assert json.loads((gen/'manifest.json').read_text())['input_sha']==sha(source)
 assert json.loads((basefile.parent/'math_manifest.json').read_text())['data_sha256']==sha(source)
 assert [r['index'] for r in base]==list(range(len(raw)))
 generated=read(gen/'predictions.jsonl');assert len(generated)==8*len(raw)
 assert {(g['index'],g['sample']) for g in generated}=={(i,j) for i in range(len(raw)) for j in range(8)}
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 known={tuple(x) for x in json.loads((S/'audits/nll_noguided_review_decisions.json').read_text())['excluded_target_keys']};old_all=read(S/'data/sample_all.jsonl');core_excluded=[r['original_index'] for r in old_all if (r['original_index'],hashlib.sha256(r['solution'].encode()).hexdigest()) in known];old=[r for r in old_all if r['original_index'] not in core_excluded]
 decision=S/'audits/sample_expansion_review_decisions.json';excluded=set(json.loads(decision.read_text())['excluded_expansion_indices']) if decision.exists() else set();pool=defaultdict(list);reasons=Counter()
 for g in generated:
  i=g['index'];r=raw[i];text=g['prediction'];assert r['problem']==g['problem']
  if g['finish_reason']!='stop':reasons['incomplete']+=1;continue
  boxes=boxed_contents(text)
  if not boxes or not verify(parse(r['solution']),parse('\\boxed{'+boxes[-1]+'}')):reasons['incorrect_last_box']+=1;continue
  n=len(tok.encode(text,add_special_tokens=False))+1;prompt=len(chat_prompt_ids(tok,r['problem']))
  if prompt+max(n,len(tok.encode(r['solution'],add_special_tokens=False))+1)>2048:reasons['too_long']+=1;continue
  if any(t in text for t in ['[asy]','\\begin{asy}','```asy','<think>','</think>']):reasons['code_or_thinking']+=1;continue
  rr=dict(r,solution=text,expansion_index=i,generation_sample=g['sample'],target_source='sample_expansion',target_tokens=n);rr.pop('original_index',None);pool[i].append(rr)
 automatic=len(pool);chosen={i:random.Random(20261009+i).choice(v) for i,v in sorted(pool.items()) if i not in excluded};new=list(chosen.values());combined=old+new
 repeat=[dict(old[i%len(old)],repeat_slot=i,target_source='sample_expansion_exposure_control') for i in range(len(combined))];rawnew=[dict(raw[i],expansion_index=i,target_source='raw_expansion',target_tokens=len(tok.encode(raw[i]['solution'],add_special_tokens=False))+1) for i in chosen]
 sets=dict(sample_expansion_new=new,sample_expansion_combined=combined,sample_expansion_repeat_control=repeat,sample_expansion_rawnew_control=old+rawnew)
 otherpath=S/'data/expansion32_combined.jsonl'
 if otherpath.exists() and (S/'audits/expansion_target_release.json').exists():
  other={r['problem']:r for r in read(otherpath)};common=[r['problem'] for r in combined if r['problem'] in other];own={r['problem']:r for r in combined};sets['sample_expansion_common_teacher32']=[own[q] for q in common];sets['teacher32_expansion_common_sample']=[other[q] for q in common]
 audit={}
 for name,rr in sets.items():
  jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(r['target_tokens'] for r in rr),levels=dict(Counter(r['level'] for r in rr)))
 groups=defaultdict(list)
 for i,r in chosen.items():
  group=('greedy_right_' if base[i]['correct'] else 'greedy_wrong_')+('diagram' if '[asy]' in r['problem'] else 'text');groups[group].append(r)
 review=S/'audits/sample_expansion_review.jsonl'
 if not review.exists():
  rng=random.Random(20261010);samples=[]
  for group,rr in sorted(groups.items()):
   for r in rng.sample(rr,min(8,len(rr))):samples.append(dict(r,review_group=group,raw_solution=raw[r['expansion_index']]['solution']))
  jsonl(review,samples)
 write(S/'audits/sample_expansion_build.json',dict(time=time.time(),automatic_new=automatic,new=len(new),new_greedy_wrong=sum(not base[i]['correct'] for i in chosen),old_core_known_error_exclusions=core_excluded,old_core_n=len(old),review_excluded=sorted(excluded),rejections=dict(reasons),datasets=audit,review_group_sizes={g:len(v) for g,v in groups.items()},status='Provisional pending32-case review and explicitrelease',random_target_seed='20261009+expansion_index',index_contract='Expansionlocalindex never used as original2000index.',control_limitations='Repeat matchesexample exposures andhorizon but not supervisedtoken counts; rawnew/sourcecontrols matchsamequestions/order, nottargetlength. Old core excludes newly proven errors identically in pairedscalecontrols; originalreleaseddatasets remain immutable.'))
 print(json.dumps(dict(new=len(new),hard=sum(not base[i]['correct'] for i in chosen),core=len(old),sizes={n:len(v) for n,v in sets.items()}),indent=2))
if __name__=='__main__':main()
