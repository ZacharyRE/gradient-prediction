"""Select bounded qualitative cases from frozen primary outputs without cherry-picking."""
import argparse,random
from common import *
from evaluation.audit_lora_sft_review import boxed_contents
p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');a=p.parse_args();pp=S/'audits/final_transition_review_plan.json'
if a.declare:
 assert not pp.exists() and not (S/'results/confirmation_plan.json').exists()
 for d in ['math_reused5000','ood_minerva','ood_olympiad','ood_svamp']:
  assert not list((S/'results/evaluation'/d).glob('*/math_predictions.jsonl'))
 write(pp,dict(time=time.time(),script_sha256=sha(__file__),seed=20261101,per_direction={'math_reused5000':2,'ood_minerva':1,'ood_olympiad':1,'ood_svamp':1},directions=['all_five_rescue','all_five_loss'],selection='Strict baseline correctness flips in the same direction for all five frozen-primary seeds; all six outputs normalstop and contain completeboxed. Uniform random sample without replacement from sorted eligibleindices, one shared fixed RNG, dataset/direction insertionorder preserved.',review='Review baseline and predeterminedseed43 fulltexts perselectedcase; store all six fulltexts/hashes. Otherfour processvalidity is NOT inferred from finalanswer agreement. Atmost10cases; missingeligiblepools reported. Qualitative existenceexamples, not prevalence/cause-estimation or a new efficacy test.',scope='Does not alter scores, train targets, selection, or primarysuccessrules.'))
 print(pp);raise SystemExit(0)
plan=json.loads(pp.read_text());assert plan['script_sha256']==sha(__file__)
fp=S/'results/confirmation_plan.json';final=json.loads(fp.read_text());family=final['families'][final['primary_family']];names=['base']+[family[str(seed)] for seed in final['seeds']];rng=random.Random(plan['seed']);cases=[];pools=[]
for dataset,k in plan['per_direction'].items():
 rows=read(S/f'data/{dataset}.jsonl');sp=S/('audits/scoring_minerva_numeric_corrected.json' if dataset=='ood_minerva' else f'audits/scoring_confirmation_{dataset}.json');strict=json.loads(sp.read_text());root=S/'results/evaluation'/dataset
 predictions={n:read(root/n/'math_predictions.jsonl') for n in names}
 for n in names:
  assert strict['models'][n]['predictions_sha256']==sha(root/n/'math_predictions.jsonl')
  assert len(predictions[n])==len(rows)
 for direction in plan['directions']:
  eligible=[]
  for i,row in enumerate(rows):
   b=bool(strict['models']['base']['strict_vector'][i]);vv=[bool(strict['models'][n]['strict_vector'][i]) for n in names[1:]]
   change=(not b and all(vv)) if direction=='all_five_rescue' else (b and not any(vv))
   normal=all(predictions[n][i]['finish_reason']=='stop' and boxed_contents(predictions[n][i]['prediction']) for n in names)
   if change and normal:eligible.append(i)
  chosen=sorted(rng.sample(eligible,min(k,len(eligible))));pools.append(dict(dataset=dataset,direction=direction,eligible=len(eligible),selected=chosen,requested=k))
  for i in chosen:
   cases.append(dict(case_id=f'{dataset}:{direction}:{i}',dataset=dataset,direction=direction,index=i,problem=rows[i]['problem'],solution=rows[i]['solution'],primary=final['primary_family'],process_review_models=['base',family['43']],outputs={n:dict(prediction=predictions[n][i]['prediction'],strict_correct=bool(strict['models'][n]['strict_vector'][i]),original_correct=bool(predictions[n][i]['correct']),sample_hash=predictions[n][i]['sample_hash'],finish_reason=predictions[n][i]['finish_reason'],generated_tokens=predictions[n][i]['generated_tokens']) for n in names}))
jsonl(S/'audits/final_transition_review.jsonl',cases)
write(S/'audits/final_transition_review_selection.json',dict(time=time.time(),plan_sha256=sha(pp),confirmation_plan_sha256=sha(fp),n_cases=len(cases),pools=pools,review_file_sha256=sha(S/'audits/final_transition_review.jsonl'),scope=plan['review']))
print(json.dumps(dict(n_cases=len(cases),pools=pools)))
