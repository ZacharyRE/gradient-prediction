"""Provisional same-question NLL and random targets from reused candidate scores."""
import argparse,random
from collections import defaultdict,Counter
from common import *
p=argparse.ArgumentParser();p.add_argument('--include-guided',action='store_true');a=p.parse_args()
scored=S/'results/candidate_nll/reused_pool';assert (scored/'complete.json').exists()
rows=read(S/'data/nll_candidate_pool.jsonl');scores=read(scored/'scores.jsonl');assert len(rows)==len(scores)
assert json.loads((scored/'manifest.json').read_text())['input_sha256']==sha(S/'data/nll_candidate_pool.jsonl')
prefix='nll_guided' if a.include_guided else 'nll_noguided';allowed_guided=set()
if a.include_guided:
 decision=json.loads((S/'audits/judge_calibration_decision.json').read_text());assert decision['use_for_guided_filter'] is True
 judge=Path(decision['judge_output']);assert (judge/'complete.json').exists()
 for r in read(judge/'predictions.jsonl'):
  if r['finish_reason']=='stop' and r['judgment'] and r['judgment']['verdict']=='sound':allowed_guided.add((r['original_index'],r['target_sha256']))
review_decisions=S/f'audits/{prefix}_review_decisions.json'
excluded={tuple(x) for x in json.loads(review_decisions.read_text())['excluded_target_keys']} if review_decisions.exists() else set()
# Mathematical counterexamples remain exclusions when adding a new source.
if a.include_guided:
 prior=S/'audits/nll_noguided_review_decisions.json'
 if prior.exists():excluded.update(tuple(x) for x in json.loads(prior.read_text())['excluded_target_keys'])
pool=defaultdict(list);sample_pool=defaultdict(list)
for r,s in zip(rows,scores):
 assert r['original_index']==s['original_index'] and r['target_sha256']==s['target_sha256']
 if (r['original_index'],r['target_sha256']) in excluded:continue
 sources=set(r['candidate_sources']);key=(r['original_index'],r['target_sha256'])
 if sources=={'guided_quarantined'} and key not in allowed_guided:continue
 rr=dict(r,nll_mean=s['nll_mean'],target_tokens=s['tokens']);pool[r['original_index']].append(rr)
 if 'sample' in sources:sample_pool[r['original_index']].append(rr)
def select(bank,mode):
 out=[]
 for i,candidates in sorted(bank.items()):
  if mode=='min':r=min(candidates,key=lambda x:(x['nll_mean'],x['target_sha256']))
  else:r=random.Random(20260920+i).choice(sorted(candidates,key=lambda x:x['target_sha256']))
  out.append(dict(r,target_source=prefix+'_'+mode))
 return out
sets={prefix+'_min':select(pool,'min'),prefix+'_random':select(pool,'random'),prefix+'_sample_min':select(sample_pool,'min')}
assert [r['problem'] for r in sets[prefix+'_min']]==[r['problem'] for r in sets[prefix+'_random']]
audit={}
for name,rr in sets.items():
 jsonl(S/f'data/{name}.jsonl',rr);audit[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(r['target_tokens'] for r in rr),mean_nll=sum(r['nll_mean'] for r in rr)/len(rr),source_memberships=dict(Counter(s for r in rr for s in set(r['candidate_sources']))))
base=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl');sampleids={r['original_index'] for r in read(S/'data/sample_all.jsonl')};raw=read(S/'data/train.jsonl')
review_path=S/f'audits/{prefix}_review.jsonl'
if not review_path.exists():
 review=[];rng=random.Random(20260921)
 for name in [prefix+'_min',prefix+'_random']:
  groups=defaultdict(list)
  for r in sets[name]:
   i=r['original_index'];group='greedy_right' if base[i]['correct'] else 'greedy_wrong_sample_covered' if i in sampleids else 'greedy_wrong_sample_uncovered';groups[group].append(r)
  for group,rr in sorted(groups.items()):
   for r in rng.sample(rr,min(8,len(rr))):review.append(dict(r,review_dataset=name,review_group=group,raw_solution=raw[r['original_index']]['solution']))
 jsonl(review_path,review)
write(S/f'audits/{prefix}_build.json',dict(time=time.time(),datasets=audit,excluded_target_keys=sorted(excluded),include_guided=a.include_guided,status='Provisional until assistant review and explicit release; NLL is not a process certificate.',same_question_random_control=True,random_seed_per_question='20260920+original_index',sources_sha256=sha(S/'data/nll_candidate_pool.jsonl'),scores_sha256=sha(scored/'scores.jsonl')))
print(json.dumps(audit,indent=2))
