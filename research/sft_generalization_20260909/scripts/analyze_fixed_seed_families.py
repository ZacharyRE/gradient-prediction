"""Fixed development endpoints for all three seeds, including predeclared epoch averages."""
import argparse
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['dev','math_reused1500'],default='dev');p.add_argument('--scoring-audits',nargs='*',type=Path);args=p.parse_args()
root=S/'results/evaluation'/args.dataset
if not (root/'base/math_summary.json').exists():print('Pending baseline',args.dataset);raise SystemExit(0)
base=read(root/'base/math_predictions.jsonl');bc=np.array([r['correct'] for r in base],dtype=float)
strict={}
for audit in args.scoring_audits or []:
 obj=json.loads(audit.read_text());assert obj['dataset_sha256']==sha(S/f'data/{args.dataset}.jsonl')
 for name,record in obj['models'].items():
  assert record['predictions_sha256']==sha(root/name/'math_predictions.jsonl')
  if name in strict:assert strict[name]==record
  strict[name]=record
if args.scoring_audits:
 assert 'base' in strict
 bc=np.array(strict['base']['strict_vector'],dtype=float);assert len(bc)==len(base)
if args.dataset=='math_reused1500':
 plan=json.loads((S/'audits/math_reused1500_fixed_families_plan.json').read_text());assert sha(S/'data/math_reused1500.jsonl')==plan['data_sha256']
families={'sample_epoch2':('sample_all_lr1e5_s{seed}_epoch2',2),'teacher7_epoch4':('teacher_all_lr5e5_s{seed}_epoch4',4),'sample_avg12':('avg_sample12_s{seed}_epoch2',2),'teacher7_avg1234':('avg_teacher7_1234_s{seed}_epoch4',4)};out={}
for family,(template,epoch) in families.items():
 result=dict(fixed_epoch=epoch,per_seed=[],missing=[]);matrix=[]
 for seed in [43,44,45]:
  name=template.format(seed=seed);p=root/name/'math_predictions.jsonl'
  if not (p.parent/'math_summary.json').exists() or (args.scoring_audits and name not in strict):result['missing'].append(seed);continue
  rr=read(p);assert [r['sample_hash'] for r in rr]==[r['sample_hash'] for r in base];x=np.array([r['correct'] for r in rr],dtype=float);d=x-bc;matrix.append(d)
  if args.scoring_audits:
   x=np.array(strict[name]['strict_vector'],dtype=float);assert len(x)==len(base)
   d=x-bc;matrix[-1]=d
  if args.dataset=='math_reused1500':
   manifest=json.loads((p.parent/'math_manifest.json').read_text());assert manifest['adapter']['files']==plan['adapter_files_sha256'][name]
  result['per_seed'].append(dict(seed=seed,model=name,correct=int(x.sum()),delta_pp=float(d.mean()*100),wins=int((d>0).sum()),losses=int((d<0).sum())))
 if not result['missing']:
  a=np.array(matrix);rng=np.random.default_rng(20261004);boots=[];sq=[]
  for _ in range(10000):
   ix=rng.integers(len(base),size=len(base));ss=rng.integers(3,size=3);boots.append(a[:,ix].mean()*100);sq.append(a[ss][:,ix].mean()*100)
  result.update(all_seeds_positive=all(r['delta_pp']>0 for r in result['per_seed']),mean_delta_pp=float(a.mean()*100),question_ci95_pp=np.quantile(boots,[.025,.975]).tolist(),seed_question_ci95_pp=np.quantile(sq,[.025,.975]).tolist())
  tail=.05/(2*len(families))
  result['four_family_bonferroni_question_ci95_pp']=np.quantile(boots,[tail,1-tail]).tolist()
  result['four_family_bonferroni_seed_question_ci95_pp']=np.quantile(sq,[tail,1-tail]).tolist()
  result['multiplicity_scope']='Bonferroni percentile intervals over these four fixed families only; does not correct the entire adaptive research search.'
 out[family]=result
destination=S/'audits'/('fixed_seed_development.json' if args.dataset=='dev' else 'fixed_seed_math_reused1500.json')
if args.scoring_audits:destination=destination.with_stem(destination.stem+'_strict')
write(destination,dict(time=time.time(),dataset=args.dataset,base_correct=int(bc.sum()),n=len(base),families=out,scoring_audits={str(p):sha(p) for p in args.scoring_audits or []},limitation='Development stability or explicitly historically reused validation only; endpoints frozen in respective plans after seed43 exploration. Multiple family searches remain exploratory; three-seed bootstrap limited support. No general efficacy without reserved OOD. Strict last-box, when supplied, is sensitivity and does not replace the primary score.'))
print(json.dumps(out,indent=2))
