"""Fixed development endpoints for coverage, horizon, markup and second-LR NLL controls."""
import numpy as np
from common import *
auditpath=S/'audits/scoring_midnight_fixed_controls.json'
while not auditpath.exists() and time.time()<DEADLINE and not (S/'STOP_DIAGNOSTICS').exists():time.sleep(2)
assert auditpath.exists();audit=json.loads(auditpath.read_text());root=S/'results/evaluation/dev';raw={};strict={}
for name,r in audit['models'].items():
 p=root/name/'math_predictions.jsonl';assert sha(p)==r['predictions_sha256'];rows=read(p)
 raw[name]=np.array([x['correct'] for x in rows],int);strict[name]=np.array(r['strict_vector'],int)
 assert len(rows)==500
pairs=[
 ('sample_expansion_combined_lr1e5_s43_epoch2','sample_expansion_repeat_control_lr1e5_s43_epoch2','Same4149 training slots: expanded questions versus repeating cleaned old core'),
 ('sample_all_shortcos2_lr1e5_s43_epoch2','sample_all_lr1e5_s43_epoch2','Same2-epoch exposure: completed2-epoch cosine versus original8-epoch horizon'),
 ('teacher_all_shortcos4_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4','Same4-epoch exposure: completed4-epoch cosine versus original8-epoch horizon'),
 ('teacher32_plain_markup_lr1e5_s43_epoch4','teacher32_all_qkvo16_lr1e5_s43_epoch4','Cosmetic markdown removal atLR1e-5; math/words/numbers retained, tokenization and length change'),
 ('teacher32_plain_markup_lr5e5_s43_epoch4','teacher32_all_qkvo16_lr5e5_s43_epoch4','Cosmetic markdown removal atLR5e-5; math/words/numbers retained, tokenization and length change'),
 ('nll_noguided_min_lr5e5_s43_epoch2','nll_noguided_random_lr5e5_s43_epoch2','SecondLR5e-5 same1758 questions: within-question minimum NLL versus random retained target')]
def compare(d):
 rng=np.random.default_rng(20260909);boots=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*100
 return dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boots,[.025,.975]).tolist(),wins=int((d>0).sum()),losses=int((d<0).sum()))
result={}
for label,vectors in [('raw',raw),('strict',strict)]:
 result[label]=dict(base_correct=int(vectors['base'].sum()),models={n:dict(correct=int(v.sum()),vs_base=compare(v-vectors['base'])) for n,v in vectors.items()},pairs=[dict(treatment=t,control=c,hypothesis=h,treatment_correct=int(vectors[t].sum()),control_correct=int(vectors[c].sum()),**compare(vectors[t]-vectors[c])) for t,c,h in pairs])
write(S/'results/midnight_fixed_control_analysis.json',dict(time=time.time(),script_sha256=sha(__file__),scoring_audit_sha256=sha(auditpath),n=500,seed=43,results=result,scope='Six selected-development paired hypotheses, fixed endpoints, all confidence intervals unadjusted. Two LRs are not independent training seeds; neither partial improvement over a harmful comparator nor a nonsignificant difference proves general efficacy/inefficacy. Global raw Holm sensitivity is kept separately in development_pairs.json.'))
print(json.dumps({k:v['pairs'] for k,v in result.items()},indent=2))
