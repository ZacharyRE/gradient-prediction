"""Bind previously read full outputs to unchanged authoritative unit inventory."""
from common import *
pf=S/'audits/minerva_explicit_unit_preview_cases.jsonl';preview=read(pf)
rp=S/'audits/minerva_unit_preview_review.json';r=json.loads(rp.read_text());assert r['all_preview_full_outputs_read'] and r['case_file_sha256']==sha(pf)
af=S/'audits/minerva_explicit_unit_cases.jsonl';actual=read(af);ip=S/'audits/minerva_explicit_unit_inventory.json';inv=json.loads(ip.read_text())
assert inv['case_file_sha256']==sha(af) and len(actual)==len(preview)==8 and inv['n_models']==21
pinv=json.loads((S/'audits/minerva_explicit_unit_preview_inventory.json').read_text())
for k in ['identities','per_model_counts','unique_question_box_cases','unique_question_indices']:assert inv[k]==pinv[k],k
false_negatives=[]
for c,p,d in zip(actual,preview,r['decisions']):
 assert c['case_id']==p['case_id']==d['case_id']
 for key in ['index','problem','reference_answer','solution','last_box']:assert c[key]==p[key]
 assert len(c['outputs'])==len(p['outputs'])==len(d['output_bindings'])
 for o,old,b in zip(c['outputs'],p['outputs'],d['output_bindings']):
  for k in old:assert o[k]==old[k],(c['case_id'],k)
  assert b['model']==o['model'] and b['predictions_sha256']==o['predictions_sha256'] and b['text_sha256']==hashlib.sha256(o['prediction'].encode()).hexdigest()
  b.update(numeric_primary_correct=o['numeric_primary_correct'],numeric_strict_correct=o['numeric_strict_correct'])
  if d['final_answer_correct'] and not o['numeric_primary_correct']:
   false_negatives.append(dict(index=c['index'],model=o['model'],last_box=c['last_box']))
assert len(false_negatives)==3
r.update(time=time.time(),preview_only=False,all_reviewed=True,authoritative_reconciliation_pending=False,case_file_sha256=sha(af),inventory_sha256=sha(ip),actual_numeric_false_negatives=false_negatives)
r['report_markdown']='按任何Minerva输出前声明的规则，完整检查21模型全部最后答案框中的百分号/角度标记，共8种题目与答案组合、11份实际输出。3份最终答案正确却被数值解析拒绝：主方案seed45的100%（index231）和12.5%（242），以及扩容未平均seed44的12.5%（242）。三份推导均含明确错误，所以只认最终答案，不认整个推导有效；独立计算分别为一阶反应k≈0.187977/s、600s后完成率四舍五入100%，以及平衡点排斥/吸引能量比1/n=1/8。其余8份输出并非单纯单位等价，例如90度错把折射角当临界入射角、87.5%取了所求比值的补数、12.4%超过冻结1e−4容差。index212的circ是标准态焓符号而非角度，模型也没有给出确定的能量。该题12340°C的字面温度高于[Ir熔点2446°C](https://periodic-table.rsc.org/element/77/iridium)；假设原意1234°C可重建约1.50eV，但OCR修正没有被证实，数据未改动。index8的2%遮挡不能直接当星等变化；按[NASA给出的对数星等关系](https://imagine.gsfc.nasa.gov/ask_astro/night_sky.html)，变化幅度为−2.5log10(.98)≈.0219348。实际评分遗漏只在独立的事后敏感性中修正，冻结数值标准不变。本检查不覆盖所有单位换算或仅在框外写单位的答案。'
r['source_sha256']={str(p.relative_to(S)):sha(p) for p in [pf,rp,af,ip,S/'audits/minerva_explicit_unit_preview_inventory.json',S/'audits/minerva_unit_case_audit_plan.json']}
write(S/'audits/minerva_explicit_unit_review.json',r);print(json.dumps(false_negatives))
