"""Reused candidate bank; no new generation or evaluation-set access."""
from collections import Counter
from common import *
from transformers import AutoTokenizer
from training.train_lora_sft import chat_prompt_ids
from evaluation.audit_lora_sft_review import boxed_contents
from math_verify import parse,verify
raw=read(S/'data/train.jsonl');byq={r['problem']:i for i,r in enumerate(raw)}
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
sample_ids={r['original_index'] for r in read(S/'data/sample_all.jsonl')}
trace_exclusions={(r['kind'],r['index'],r['sample']) for r in json.loads((S/'audits/process_exclusions.json').read_text())}
pool={};rejections=Counter();prompt_lengths={i:len(chat_prompt_ids(tok,raw[i]['problem'])) for i in sample_ids}
def add(r,source):
 i=byq[r['problem']];key=(i,hashlib.sha256(r['solution'].encode()).hexdigest())
 if key in pool:pool[key]['candidate_sources'].append(source);return
 pool[key]=dict(r,original_index=i,target_source=source,candidate_sources=[source],target_sha256=key[1])
for g in read(S/'results/generation/sample/predictions.jsonl'):
 i=g['index']
 if i not in sample_ids:continue
 if ('sample',i,g['sample']) in trace_exclusions:continue
 if g['finish_reason']!='stop':continue
 text=g['prediction'];boxes=boxed_contents(text)
 if not boxes or not verify(parse(raw[i]['solution']),parse('\\boxed{'+boxes[-1]+'}')):continue
 n=len(tok.encode(text,add_special_tokens=False))+1
 if n+prompt_lengths[i]>2048 or '[asy]' in text or '\\begin{asy}' in text:continue
 add(dict(raw[i],solution=text,generation_sample=g['sample']), 'sample')
for name,source in [('previous_self','greedy'),('teacher_all','teacher7'),('teacher32_all','teacher32'),('guided_all','guided_quarantined')]:
 for r in read(S/f'data/{name}.jsonl'):add(r,source)
rows=sorted(pool.values(),key=lambda r:(r['original_index'],r['target_sha256']))
jsonl(S/'data/nll_candidate_pool.jsonl',rows)
write(S/'audits/nll_candidate_plan.json',dict(time=time.time(),n=len(rows),questions=len({r['original_index'] for r in rows}),source_counts=dict(Counter(s for r in rows for s in set(r['candidate_sources']))),input_sha256=sha(S/'data/nll_candidate_pool.jsonl'),hypothesis='Within-question minimum student mean token NLL selects more compatible correct targets; compare same-question random/source controls. Guided candidates remain quarantined until judge calibration and process filtering. NLL is not correctness or process quality.',selection_status='No targets selected, no training queued. Base forward scoring only; no dev or OOD outcomes used in candidate filtering.'))
print('NLL candidate rows',len(rows),flush=True)
