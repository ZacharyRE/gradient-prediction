"""Bind explicitly reviewed full outputs to all final OOD prediction/score files."""
from common import *
from collections import Counter
plan=json.loads((S/'audits/ood_scoring_disagreement_review_plan.json').read_text())
preview=read(S/'audits/ood_scoring_disagreement_preview_cases.jsonl')
review=json.loads((S/'audits/ood_scoring_disagreement_preview_review.json').read_text())
assert review['all_preview_outputs_reviewed'] and sha(S/'audits/ood_scoring_disagreement_preview_cases.jsonl')==review['case_file_sha256']
addon=json.loads((S/'audits/ood_scoring_disagreement_amc_addendum.json').read_text());assert addon['all_additional_outputs_reviewed']
preview+=addon['cases'];preview.sort(key=lambda c:(c['dataset'],c['index'],c['model']))
review['decisions']+=addon['decisions'];review['decisions'].sort(key=lambda c:(c['dataset'],c['index'],c['model']))
review['counts']=dict(Counter(str(d['final_answer_correct']) for d in review['decisions']));review['n_cases']=len(preview)
cases=[];bindings={};coverage={}
refs={r['index'] for r in json.loads((S/'audits/minerva_numeric_contract.json').read_text())['numeric_references']}
for ds in plan['datasets']:
 rows=read(S/'data'/f'{ds}.jsonl');seen=set()
 for kind in ['confirmation','supplement']:
  ap=S/f'audits/scoring_{kind}_{ds}.json';audit=json.loads(ap.read_text());bindings[str(ap.relative_to(S))]=sha(ap)
  for name,m in audit['models'].items():
   assert name not in seen;seen.add(name)
   p=S/'results/evaluation'/ds/name/'math_predictions.jsonl';pred=read(p);assert sha(p)==m['predictions_sha256'] and len(pred)==len(rows)
   for i,(o,st) in enumerate(zip(pred,m['strict_vector'])):
    if ds=='ood_minerva' and i in refs:continue
    if bool(o['correct'])==bool(st):continue
    cases.append(dict(case_id=f'{ds}:{name}:{i}',dataset=ds,model=name,index=i,problem=rows[i]['problem'],reference_answer=rows[i]['answer'],solution=rows[i]['solution'],prediction=o['prediction'],original_correct=bool(o['correct']),strict_correct=bool(st),predictions_sha256=sha(p),finish_reason=o.get('finish_reason')))
 assert len(seen)==21;coverage[ds]=len(seen)
cases.sort(key=lambda c:(c['dataset'],c['index'],c['model']))
if cases!=preview:
 old={c['case_id']:c for c in preview};new={c['case_id']:c for c in cases}
 print('ADDED', sorted(set(new)-set(old)), 'REMOVED',sorted(set(old)-set(new)))
 print('CHANGED',[(k,[z for z in new[k] if new[k][z]!=old[k].get(z)]) for k in set(old)&set(new) if old[k]!=new[k]])
 jsonl(S/'audits/ood_scoring_disagreement_final_pending_cases.jsonl',cases)
 raise AssertionError('New/changed cases require actual review; do not autoapprove')
assert [x['case_id'] for x in review['decisions']]==[c['case_id'] for c in cases]
for c,d in zip(cases,review['decisions']):
 assert d['full_text_read'] and d['text_sha256']==hashlib.sha256(c['prediction'].encode()).hexdigest() and d['predictions_sha256']==c['predictions_sha256']
jsonl(S/'audits/ood_scoring_disagreement_cases.jsonl',cases)
write(S/'audits/ood_scoring_disagreement_inventory.json',dict(time=time.time(),all_four_datasets_all21_models_complete=True,coverage=coverage,n_cases=len(cases),per_dataset=dict(Counter(c['dataset'] for c in cases)),case_file_sha256=sha(S/'audits/ood_scoring_disagreement_cases.jsonl'),plan_sha256=sha(S/'audits/ood_scoring_disagreement_review_plan.json'),source_sha256=bindings,scope=plan['scope']))
review['scope']='All74full outputs reviewed, including one identical full-text duplicate. All original scores preserved; four unresolved cases bounded; agreed outputs not certified.'
review.update(time=time.time(),preview_only=False,all_reviewed=True,authoritative_reconciliation_pending=False,final_inventory_sha256=sha(S/'audits/ood_scoring_disagreement_inventory.json'),case_file_sha256=sha(S/'audits/ood_scoring_disagreement_cases.jsonl'))
review['report_markdown']='全四个OOD、21个固定模型的raw/strict分歧共74个输出，已逐一读取完整文本并独立检查算术/推导；61个最终答案可确认正确、9个错误、4个因题面或未完成回答而保留不确定。SVAMP有56个baseline和2个补充模型输出被严格评分拒绝：56个baseline中54个明确正确但没有box，另2个存在题面时序问题；两个补充输出也正确且无box。主候选五个seed在SVAMP的raw/strict分数完全相同，因此严格OOD正增益主要来自对baseline的格式惩罚。Minerva有5个正确的线性期望表达式被原抽取拒绝，但strict还错误接受一个平方能量表达式；Olympiad也存在strict误接受含错误成员的答案列表。不存在一种解析器在全部实际案例上都可靠。人工敏感性只修正本次明确复核的分歧，其余一致判分仍未经全面认证；不改变冻结指标。'
write(S/'audits/ood_scoring_disagreement_review.json',review)
print(json.dumps(dict(n=len(cases),counts=review['counts'],coverage=coverage)))
