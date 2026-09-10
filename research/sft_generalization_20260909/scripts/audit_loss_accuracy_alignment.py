"""Read-only, fixed-pair check of reference likelihood versus free generation."""
from common import *
from gradient_geometry.sft_protocol import sample_identity
import numpy as np
pairs=[('sample_all_lr1e5_s43','sample_matchedraw_lr1e5_s43',2),('teacher_all_lr5e5_s43','teacher_matchedraw_lr5e5_s43',4)]
dev=read(S/'data/dev.jsonl');ce=read(S/'data/dev_ce.jsonl');positions={sample_identity(r):i for i,r in enumerate(dev)}
assert len(positions)==500
indices=[positions[sample_identity(r)] for r in ce];assert len(indices)==len(set(indices))==64
for r,i in zip(ce,indices):assert r==dev[i]
basepath=S/'results/evaluation/dev/base/math_predictions.jsonl';base=read(basepath);assert [r['sample_hash'] for r in base]==list(positions)
rows=[]
for generated,raw,epoch in pairs:
 group=[]
 for name in [raw,generated]:
  out=S/'results/training'/name;h=json.loads((out/'history.json').read_text());end=next(r for r in h if r.get('epoch')==epoch and 'dev_ce' in r)
  predpath=S/f'results/evaluation/dev/{name}_epoch{epoch}/math_predictions.jsonl';rr=read(predpath)
  assert [r['sample_hash'] for r in rr]==[sample_identity(r) for r in dev]
  group.append(dict(name=name,epoch=epoch,history_sha256=sha(out/'history.json'),prediction_sha256=sha(predpath),initial_raw_reference_ce=h[0]['dev_ce'],endpoint_raw_reference_ce=end['dev_ce'],raw_reference_ce_change=end['dev_ce']-h[0]['dev_ce'],correct_same64=sum(bool(rr[i]['correct']) for i in indices),base_correct_same64=sum(bool(base[i]['correct']) for i in indices),correct_dev500=sum(bool(r['correct']) for r in rr),base_correct_dev500=sum(bool(r['correct']) for r in base),probe_initial=h[0]['probe_ce'],probe_endpoint=end['probe_ce']))
 assert group[0]['initial_raw_reference_ce']==group[1]['initial_raw_reference_ce']
 rows.append(dict(raw=group[0],generated=group[1]))
write(S/'audits/loss_accuracy_alignment.json',dict(time=time.time(),script_sha256=sha(__file__),dev_ce_n=64,dev_n=500,ce_indices_in_dev=indices,dev_ce_sha256=sha(S/'data/dev_ce.jsonl'),dev_sha256=sha(S/'data/dev.jsonl'),base_predictions_sha256=sha(basepath),pairs=rows,scope='Descriptive fixed-pair audit. Loss is BF16 HF teacher-forced reference CE on64questions; accuracy is same-vLLM-protocol free generation, shown on exactlysame64 andall500. Not a purebackend comparison, crossvalidated selection test, or proof that all validation losses are misleading. Per-step train losses are NOT treated as full-training-set CE.'))
print(json.dumps(rows,indent=2))
