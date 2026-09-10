"""Favorable scoring ceilings for missing-box/truncation-only explanations."""
import random
from common import *
from evaluation.audit_lora_sft_review import boxed_contents
audit=json.loads((S/'audits/scoring_post_sampling_primary.json').read_text());root=S/'results/evaluation/dev';data=read(S/'data/dev.jsonl')
base=audit['models']['base']['strict_vector'];results=[];review=[];chosen=set()
for label,raw,source in [('student','sample_matchedraw_lr1e5_s43_epoch2','sample_all_lr1e5_s43_epoch2'),('teacher7','teacher_matchedraw_lr5e5_s43_epoch4','teacher_all_lr5e5_s43_epoch4')]:
 p=root/raw/'math_predictions.jsonl';rr=read(p);ss=read(root/source/'math_predictions.jsonl')
 assert sha(p)==audit['models'][raw]['predictions_sha256']
 rvec=audit['models'][raw]['strict_vector'];svec=audit['models'][source]['strict_vector']
 missing=[not boxed_contents(r['prediction']) for r in rr];truncated=[r['finish_reason']=='length' for r in rr];union=[a or b for a,b in zip(missing,truncated)]
 ceilings={name:sum(bool(v) or flag for v,flag in zip(rvec,flags)) for name,flags in [('missing_box',missing),('truncation',truncated),('either',union)]}
 valid_wrong=[i for i,(v,flag) in enumerate(zip(rvec,union)) if not v and not flag]
 pool=[i for i in valid_wrong if base[i] and svec[i] and i not in chosen]
 indices=random.Random(20261023+len(results)).sample(pool,min(2,len(pool)));chosen.update(indices)
 for i in indices:review.append(dict(family=label,index=i,problem=data[i]['problem'],solution=data[i]['solution'],raw_prediction=rr[i]['prediction'],generated_target_model_prediction=ss[i]['prediction'],baseline_prediction=read(root/'base/math_predictions.jsonl')[i]['prediction'],raw_model=raw,source_model=source,raw_finish_reason=rr[i]['finish_reason'],raw_last_box=boxed_contents(rr[i]['prediction'])[-1]))
 results.append(dict(family=label,raw_model=raw,source_model=source,n=len(rr),raw_strict_correct=sum(rvec),source_strict_correct=sum(svec),baseline_strict_correct=sum(base),missing_box=sum(missing),truncation=sum(truncated),union=sum(union),oracle_ceiling=ceilings,normal_complete_box_strict_wrong=len(valid_wrong),ceiling_still_below_source=ceilings['either']<sum(svec),ceiling_still_below_baseline=ceilings['either']<sum(base),selected_review_indices=indices))
write(S/'audits/raw_format_truncation_bounds.json',dict(time=time.time(),script_sha256=sha(__file__),scoring_audit_sha256=sha(S/'audits/scoring_post_sampling_primary.json'),results=results,scope='Favorable classification bound under fixed strict scorer: count every flagged output correct while leaving other outputs unchanged. This excludes only an explanation confined to these flags; changing formatting prompts or training can also change other answers, so it is not a general causal proof that formatting/budget never matter.'))
dest=S/'audits/raw_normal_boxed_review.jsonl'
if dest.exists():assert read(dest)==review
else:jsonl(dest,review)
print(json.dumps(results,indent=2))
