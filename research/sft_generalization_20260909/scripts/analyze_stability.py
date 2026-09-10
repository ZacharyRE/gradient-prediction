"""Frozen epoch2 development stability check; not a fresh benchmark claim."""
import numpy as np
from common import *
plan=json.loads((S/'audits/sample_seed_stability_plan.json').read_text());root=S/'results/evaluation/dev';base=read(root/'base/math_predictions.jsonl');bv=np.array([r['correct'] for r in base],int);rr=[];missing=[];deltas=[]
for seed in plan['seeds']:
 name=f'sample_all_lr1e5_s{seed}_epoch2';p=root/name/'math_summary.json'
 if not p.exists():missing.append(seed);continue
 rows=read(p.with_name('math_predictions.jsonl'));assert [r['sample_hash'] for r in rows]==[r['sample_hash'] for r in base]
 v=np.array([r['correct'] for r in rows],int);d=v-bv;deltas.append(d)
 rr.append(dict(seed=seed,correct=int(v.sum()),delta_pp=float(d.mean()*100),wins=int(((v==1)&(bv==0)).sum()),losses=int(((v==0)&(bv==1)).sum())))
out=dict(time=time.time(),phase=plan['phase'],selected_epoch=2,base_correct=int(bv.sum()),n=len(base),per_seed=rr,missing=missing)
if not missing:
 d=np.stack(deltas);rng=np.random.default_rng(20260922);b=[];cross=[]
 for _ in range(10000):
  ix=rng.integers(len(base),size=len(base));si=rng.integers(3,size=3);b.append(d[:,ix].mean()*100);cross.append(d[si][:,ix].mean()*100)
 out.update(every_seed_above_base=bool(np.all(d.mean(1)>0)),mean_delta_pp=float(d.mean()*100),question_ci95=np.quantile(b,[.025,.975]).tolist(),seed_question_ci95=np.quantile(cross,[.025,.975]).tolist(),limitation='Seed43 selected on this development set; seeds44/45 endpoint frozen before outcomes. No independent OOD inference.')
write(S/'results/sample_seed_stability.json',out);print(json.dumps(out,indent=2))
