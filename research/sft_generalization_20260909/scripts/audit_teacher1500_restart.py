"""Audit baseline and nonzero-adapter replays following long diagnostics/shared load."""
from common import *
root=S/'results/evaluation/dev';pairs=[('base_after_teacher1500_extension','base'),('previous_self_sdpa_after_teacher1500_extension','previous_self_sdpa_dispatch_v2_repeat')];out=[]
for new,old in pairs:
 a=root/new/'math_predictions.jsonl';b=root/old/'math_predictions.jsonl'
 assert a.with_name('math_summary.json').exists() and b.with_name('math_summary.json').exists()
 aa=read(a);bb=read(b);assert len(aa)==len(bb)==500
 keys=['sample_hash','prediction','correct','finish_reason','generated_tokens'];counts={k:sum(x[k]==y[k] for x,y in zip(aa,bb)) for k in keys}
 assert all(n==500 for n in counts.values()),(new,counts)
 ma=json.loads(a.with_name('math_manifest.json').read_text());mb=json.loads(b.with_name('math_manifest.json').read_text())
 for key in ['model','adapter','data_sha256','engine','batch_invariant','max_tokens','temperature','chunk_size']:assert ma[key]==mb[key],key
 out.append(dict(new=new,reference=old,n=500,equal_counts=counts,new_predictions_sha256=sha(a),reference_predictions_sha256=sha(b)))
write(S/'audits/teacher1500_restart_protocol_replay.json',dict(time=time.time(),passed=True,comparisons=out,scope='Specific baseline/nonzero inference paths under current shared-device conditions; does not certify every future run or every backend.'))
print(json.dumps(out))
