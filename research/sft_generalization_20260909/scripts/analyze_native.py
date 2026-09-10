"""Compare treatment effects within each backend before comparing backends."""
import numpy as np
from common import *
native=S/'results/native/dev';vllm=S/'results/evaluation/dev';manifest=json.loads((S/'results/native_recheck_manifest.json').read_text())
def interval(d):
 d=np.array(d,dtype=float);rng=np.random.default_rng(20260909);b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*100
 return dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(b,[.025,.975]).tolist())
if not (native/'base/summary.json').exists():print('Native baseline incomplete');exit()
nb=read(native/'base/predictions.jsonl');vb=read(vllm/'base/math_predictions.jsonl')[:len(nb)];assert [r['sample_hash'] for r in nb]==[r['sample_hash'] for r in vb]
nbv=np.array([r['correct'] for r in nb],int);vbv=np.array([r['correct'] for r in vb],int);rows=[];missing=[]
for name in manifest:
 if name=='base':continue
 if not (native/name/'summary.json').exists() or not (vllm/name/'math_summary.json').exists():missing.append(name);continue
 nr=read(native/name/'predictions.jsonl');vr=read(vllm/name/'math_predictions.jsonl')[:len(nb)]
 assert [r['sample_hash'] for r in nr]==[r['sample_hash'] for r in vr]==[r['sample_hash'] for r in nb]
 nv=np.array([r['correct'] for r in nr],int);vv=np.array([r['correct'] for r in vr],int)
 rows.append(dict(name=name,n=len(nb),native_correct=int(nv.sum()),vllm_correct=int(vv.sum()),native_vs_native_base=interval(nv-nbv),vllm_vs_vllm_base=interval(vv-vbv),backend_treatment_difference=interval((nv-nbv)-(vv-vbv)),exact_output_texts=sum(a['prediction']==b['prediction'] for a,b in zip(nr,vr))))
write(S/'audits/newtarget_native_comparison.json',dict(time=time.time(),native_base_correct=int(nbv.sum()),vllm_base_correct=int(vbv.sum()),n=len(nb),rows=rows,missing=missing,limitation='First128 reused dev questions, diagnostic only; backend,batch,dtype arithmetic differ jointly. Does not certify all models/ranks/checkpoints.'))
print(json.dumps(dict(rows=rows,missing=missing),indent=2))
