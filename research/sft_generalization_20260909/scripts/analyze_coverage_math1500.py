"""All six fixed scale pilots on reused1500; no seed-replication claims."""
import subprocess
import numpy as np
from common import *
planpath=S/'audits/coverage_math1500_plan.json';plan=json.loads(planpath.read_text());frozenpath=S/'audits/coverage_math1500_frozen.json';frozen=json.loads(frozenpath.read_text())
assert sha(planpath)==frozen['prospective_plan_sha256'] and sha(S/'results/coverage_math1500_manifest.json')==frozen['manifest_sha256']
root=S/'results/evaluation/math_reused1500';oldavg='avg_teacher7_1234_s43_epoch4';oldend='teacher_all_lr5e5_s43_epoch4';names=['base',oldavg,oldend,*plan['models']]
assert all((root/n/'math_summary.json').exists() for n in names)
base=read(root/'base/math_predictions.jsonl');assert len(base)==1500;raw={};strict={}
auditpath=S/'audits/scoring_coverage_math1500.json'
if not auditpath.exists():
 command=[sys.executable,str(S/'scripts/score_sensitivity.py'),'--dataset','math_reused1500','--tag','coverage_math1500','--models',*names]
 with (S/'logs/strict_coverage_math1500.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
audit=json.loads(auditpath.read_text());assert audit['dataset_sha256']==frozen['data_sha256'] and set(audit['models'])==set(names)
for name in names:
 p=root/name/'math_predictions.jsonl';rows=read(p);assert len(rows)==1500 and [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base]
 manifest=json.loads((root/name/'math_manifest.json').read_text())
 for k,v in frozen['protocol'].items():assert manifest[k]==v,(name,k)
 if name in plan['models']:assert manifest['adapter']==frozen['model_identities'][name]
 assert audit['models'][name]['predictions_sha256']==sha(p)
 raw[name]=np.array([r['correct'] for r in rows],int);strict[name]=np.array(audit['models'][name]['strict_vector'],int)
def avg(source):return 'avg_teacher7_expansion_'+source+'_1234_s43_epoch4'
def endpoint(source):return 'teacher7_expansion_'+source+'_lr5e5_s43_epoch4'
pairs=[(avg('combined'),avg('repeat_control'),'Coverage with averaging, equal4145 training slots'),(endpoint('combined'),endpoint('repeat_control'),'Coverage without averaging, equal4145 training slots'),(avg('combined'),avg('rawnew_control'),'Generated vs raw new targets on identical questions, both averaged'),(endpoint('combined'),endpoint('rawnew_control'),'Generated vs raw new targets on identical questions, both epoch4')]
for source in ['combined','repeat_control','rawnew_control']:pairs.append((avg(source),endpoint(source),'Within-run fixed epoch1..4 update average vs epoch4'))
pairs += [(avg('combined'),oldavg,'Practical expanded/cleaned-core/longer-exposure recipe vs old average; multiple factors change'),(avg('repeat_control'),oldavg,'Longer repetition exposure and5conservative core exclusions vs old average; not pure exposure-only')]
def compare(d):
 rng=np.random.default_rng(20261026);boots=np.concatenate([d[rng.integers(len(d),size=(200,len(d)))].mean(1)*100 for _ in range(50)])
 return dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boots,[.025,.975]).tolist(),wins=int((d>0).sum()),losses=int((d<0).sum()))
result={}
for label,vectors in [('raw',raw),('strict',strict)]:
 b=vectors['base'];result[label]=dict(base_correct=int(b.sum()),models={n:dict(correct=int(v.sum()),vs_base=compare(v-b)) for n,v in vectors.items()},pairs=[dict(treatment=t,control=c,hypothesis=h,treatment_correct=int(vectors[t].sum()),control_correct=int(vectors[c].sum()),**compare(vectors[t]-vectors[c])) for t,c,h in pairs])
write(S/'audits/coverage_math1500_analysis.json',dict(time=time.time(),script_sha256=sha(__file__),prospective_plan_sha256=sha(planpath),frozen_sha256=sha(frozenpath),scoring_audit_sha256=sha(auditpath),seed=43,n=1500,results=result,scope='Historically reused selected-development validation, single seed43, all six pilots retained, intervals unadjusted. Does not establish multi-seed efficacy. Same-source scale controls match slots/horizon but not all target-token counts.'))
print(json.dumps({k:dict(models={n:x['correct'] for n,x in v['models'].items()},pairs=v['pairs']) for k,v in result.items()},indent=2))
