"""Matched decoding-budget intervention, including prefix-stability checks."""
import numpy as np
from common import *

def stats(delta):
 d=np.asarray(delta,dtype=float);rng=np.random.default_rng(20261013);boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)*100
 return dict(delta_pp=float(d.mean()*100),question_ci95_pp=np.quantile(boot,[.025,.975]).tolist())

def main():
 plan=json.loads((S/'audits/long_decode_plan.json').read_text());short=S/'results/evaluation/dev';long=S/'results/evaluation_long/dev'
 if not (long/'complete.json').exists():print('Pending complete long-budget probe');return
 sources={};out={}
 for name in plan['models']:
  a=read(short/name/'math_predictions.jsonl');b=read(long/name/'math_predictions.jsonl');assert len(a)==len(b)==500 and [r['sample_hash'] for r in a]==[r['sample_hash'] for r in b]
  sa=np.array([r['correct'] for r in a],int);sb=np.array([r['correct'] for r in b],int);sources[name]=(sa,sb)
  stopped=[(x,y) for x,y in zip(a,b) if x['finish_reason']=='stop'];truncated=[(x,y) for x,y in zip(a,b) if x['finish_reason']=='length']
  out[name]=dict(short_correct=int(sa.sum()),long_correct=int(sb.sum()),own_budget_change=stats(sb-sa),short_normal_stop_n=len(stopped),short_normal_stop_identical_full_text=sum(x['prediction']==y['prediction'] for x,y in stopped),short_truncated_n=len(truncated),short_truncated_matching_text_prefix=sum(y['prediction'].startswith(x['prediction'].rstrip()) for x,y in truncated),short_truncated_wrong_to_long_correct=sum(not x['correct'] and y['correct'] for x,y in truncated),long_truncated_n=sum(x['finish_reason']=='length' for x in b))
 base_short,base_long=sources['base']
 for name,(a,b) in sources.items():
  if name=='base':continue
  out[name]['short_vs_short_baseline']=stats(a-base_short);out[name]['long_vs_long_baseline']=stats(b-base_long);out[name]['budget_by_sft_interaction']=stats((b-base_long)-(a-base_short))
 short_audit=S/'audits/scoring_long_probe_short.json';long_audit=S/'audits/scoring_long_probe_long.json'
 if short_audit.exists() and long_audit.exists():
  ss=json.loads(short_audit.read_text());ll=json.loads(long_audit.read_text())
  assert ss['dataset_sha256']==ll['dataset_sha256']==plan['data_sha256']
  ba=np.array(ss['models']['base']['strict_vector']);bb=np.array(ll['models']['base']['strict_vector'])
  for name in sources:
   for audit,root in [(ss,short),(ll,long)]:assert audit['models'][name]['predictions_sha256']==sha(root/name/'math_predictions.jsonl')
   a=np.array(ss['models'][name]['strict_vector']);b=np.array(ll['models'][name]['strict_vector'])
   out[name]['strict_last_box_sensitivity']=dict(short_correct=int(a.sum()),long_correct=int(b.sum()),own_budget_change=stats(b-a),short_vs_short_baseline=stats(a-ba),long_vs_long_baseline=stats(b-bb),budget_by_sft_interaction=stats((b-bb)-(a-ba)))
 write(S/'results/long_budget_analysis.json',dict(time=time.time(),models=out,scope='Exploratory fixed500 dev and one seed per recipe. Baseline gets the same4096-token budget. Prefix/text invariance checks whether non-budget numerical changes are visible. Effect-interaction intervals are not adjusted for multiple diagnostics; no OOD or general multi-seed claim.'))
 print(json.dumps(out,indent=2))

if __name__=='__main__':main()
