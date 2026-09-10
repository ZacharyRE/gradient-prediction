import random
from collections import Counter,defaultdict
from common import *
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify
def main():
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);raw=read(S/'data/train.jsonl');previous=read(S/'data/previous_self.jsonl');previous_q={r['problem']:r for r in previous}
 base=read(FOLLOW/'results/evaluation_bf16/train/base_self_targets/math_predictions.jsonl')
 excluded={r['original_index'] for r in json.loads((FOLLOW/'data/self_targets/review_and_final_manifest.json').read_text())['decisions'] if r['decision']=='exclude_pair'}
 ep=S/'audits/process_exclusions.json'
 trace_exclusions={(r['kind'],r['index'],r['sample']) for r in json.loads(ep.read_text())} if ep.exists() else set()
 pools={};all_candidates={};audits={};rng=random.Random(20260910)
 for kind in ['sample','teacher']:
  root=S/'results/generation'/kind;assert (root/'complete.json').exists();rr=read(root/'predictions.jsonl');accepted=defaultdict(list);reasons=Counter()
  for g in rr:
   i=g['index'];r=raw[i];assert r['problem']==g['problem'];text=g['prediction'];boxes=boxed_contents(text)
   if i in excluded:reasons['previous_process_exclusion']+=1;continue
   if (kind,i,g['sample']) in trace_exclusions:reasons['process_review_excluded_trace']+=1;continue
   if g['finish_reason']!='stop':reasons['incomplete']+=1;continue
   if not boxes or not verify(parse(r['solution']),parse('\\boxed{'+boxes[-1]+'}')):reasons['incorrect_last_box']+=1;continue
   n=len(tok.encode(text,add_special_tokens=False))+1;prompt=len(chat_prompt_ids(tok,r['problem']))
   if n+prompt>2048 or len(tok.encode(r['solution'],add_special_tokens=False))+1+prompt>2048:reasons['too_long']+=1;continue
   if '[asy]' in text or '\\begin{asy}' in text:reasons['diagram_target']+=1;continue
   accepted[i].append(dict(r,solution=text,original_index=i,generation_sample=g['sample'],target_source=kind,target_tokens=n))
  # One randomly selected verified target per problem: avoid incidental easy-question multiplicity.
  all_candidates[kind]=accepted
  pools[kind]={i:rng.choice(v) for i,v in sorted(accepted.items())};audits[kind]=dict(total_generations=len(rr),accepted_generations=sum(map(len,accepted.values())),covered_questions=len(accepted),rejections=dict(reasons),coverage_by_level=dict(Counter(raw[i]['level'] for i in accepted)))
 datasets={}
 for kind,pool in pools.items():
  datasets[kind+'_all']=list(pool.values());datasets[kind+'_raw']=[dict(raw[i],original_index=i) for i in pool]
  datasets[kind+'_easy']=[v for i,v in pool.items() if raw[i]['problem'] in previous_q]
  datasets[kind+'_hard']=[v for i,v in pool.items() if not base[i]['correct']]
  datasets[kind+'_newly_covered']=[v for i,v in pool.items() if raw[i]['problem'] not in previous_q]
  datasets[kind+'_matched_greedy']=[dict(previous_q[v['problem']],original_index=v['original_index']) for v in datasets[kind+'_easy']]
 common=sorted(set(pools['sample'])&set(pools['teacher']))
 for kind in pools:datasets[kind+'_common']=[pools[kind][i] for i in common]
 datasets['sample_multi']=[];datasets['sample_repeat']=[]
 for i in pools['sample']:
  unique=list({v['solution']:v for v in all_candidates['sample'][i]}.values());rng.shuffle(unique)
  for j in range(4):
   datasets['sample_multi'].append(unique[j%len(unique)])
   datasets['sample_repeat'].append(pools['sample'][i])
 # Preserve every previous self target and add only newly covered questions. Matched budget controls follow from this exact list.
 for kind,pool in pools.items():
  datasets[kind+'_augmented']=previous+[v for i,v in pool.items() if raw[i]['problem'] not in previous_q]
  added=[i for i,v in pool.items() if raw[i]['problem'] not in previous_q]
  datasets[kind+'_augmented_rawhard']=previous+[dict(raw[i],original_index=i) for i in added]
  datasets[kind+'_greedy_budget']=previous+[dict(previous[j%len(previous)]) for j in range(len(added))]
 review_path=S/'audits/target_review_decisions.json'
 pair_drop=set(json.loads(review_path.read_text())['excluded_original_indices']) if review_path.exists() else set()
 audits['new_review_pair_exclusions']=sorted(pair_drop)
 datasets={name:[r for r in rr if r.get('original_index') not in pair_drop] for name,rr in datasets.items()}
 for kind in pools:datasets[kind+'_greedy_budget']=datasets[kind+'_greedy_budget'][:len(datasets[kind+'_augmented'])]
 for name,rr in datasets.items():
  jsonl(S/f'data/{name}.jsonl',rr)
  audits[name]=dict(n=len(rr),sha256=sha(S/f'data/{name}.jsonl'),tokens=sum(len(tok.encode(r['solution'],add_special_tokens=False))+1 for r in rr))
 # Review stratified random targets, deliberately include newly covered problems; do not use assessment outcomes.
 review=[]
 for name in ['sample_easy','sample_hard','teacher_hard']:
  rr=datasets[name]
  for r in rng.sample(rr,min(8,len(rr))):review.append(dict(group=name,**r,raw_solution=raw[r['original_index']]['solution']))
 if not (S/'audits/target_review.jsonl').exists():jsonl(S/'audits/target_review.jsonl',review)
 write(S/'audits/target_build.json',audits);print(json.dumps(audits,indent=2))
if __name__=='__main__':main()
