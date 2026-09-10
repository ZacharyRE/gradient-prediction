"""Standalone completed mathematical scoring audit, independent of ongoing GPU work."""
from common import *
sources={}
def get(p):
 sources[p]=sha(S/p); return json.loads((S/p).read_text())
def table(head,rows):
 return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def ci(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'
manual=get('results/math_manual_disagreement_sensitivity.json')
follow=get('results/math4060_followup_sensitivity.json')
review=get('audits/math_scoring_disagreement_review_decisions.json')
operator=get('audits/math4060_operator_gold_review.json')
revision=get('audits/math_operator_semantics_revision.json')
assert review['all_reviewed'] and manual['all47_reviewed'] and operator['all_reviewed']
for r in [manual,follow]:
 for p,h in r['source_sha256'].items():assert sha(S/p)==h,p
rows=[]
for n,v in manual['counts'].items():
 rows.append(['Baseline' if n=='base' else 'SFT seed'+n.split('_s')[-1].split('_')[0],v['disagreements'],v['correct'],v['incorrect'],v['unresolved']])
text='''# 完整MATH的评分分歧审计

这是已完成的局部评分审计。主方案五seed完整MATH的原始均值差为+0.804pp，但配对区间跨零。读完全部raw/strict分歧并保留答案语义歧义后，这项结论仍未改变；不能通过事后换评分器宣称成功。

## 审查范围

在结果已知后，固定检查baseline及主方案五seed的全部47条评分分歧，覆盖35道题。复核由本助手执行，读完整题目、参考和实际输出并给出独立计算或反例；不是独立人类盲评。正确最终答案与有效推导分别判断；其余两种评分一致的输出未获全面数学认证。表中分歧数量不能当作所有题目的错误率。

'''+table(['模型','分歧数','答案正确','答案错误','保留歧义'],rows)+'''

例如，baseline在部分基础题已写出正确无框答案，却被严格答案框规则判错；一些SFT输出重复推导中的数字，原始解析器将它当作答案而判对。行列式题index4625的真实矩阵为奇数阶反对称矩阵，行列式必为零；某SFT输出因符号错误重复2(c−a)(c−b)(b−a)，取a=0,b=1,c=2即得4，所以不能因为原矩阵中出现0就判对。

## 方框算子题的语义修订

题目index4060将方框定义为约数个数算子d，目标值d(d(11)×d(20))=d(12)=6。通常推理提示又要求把最终答案放在方框内。全部21个模型有7份最终boxed12：将方框视为答案标记时它是错误12，将其视为题目算子时它却等于正确6；错误过程不自动排除最终表达式在后一解释下等价。其余14份boxed8、boxed60或boxed66在两种解释下都错，因为d(8)=4、d(60)=12、d(66)=8。

最初将21份一概判错的审查过强，已保留旧文档及结果并作[明确语义修订](audits/math_operator_semantics_revision.json)。47条审查中的4份主方案boxed12改为保留歧义，加上原先4份未完成或相互矛盾的输出，共8条。单独的47条敏感性允许每条歧义独立取不利/有利值；更连贯的追加分析对baseline和全部模型统一采用同一种方框约定，再对其余4条歧义求界。

## 全部配对敏感性结果

计数依次为seed43/44/45/46/47。题目保持配对，baseline在五个模型之间共享；第二类区间另外重采样训练seed。每个边界从逐题向量使用冻结的compare函数重算10,000次bootstrap，没有将点估计变化直接平移整个区间。

'''
for label,values in [('仅47分歧，独立歧义边界',manual['results']),*[(('统一答案框标记' if k=='answer_markup' else '统一字面约数算子'),v) for k,v in follow['results_by_interpretation'].items()]]:
 rows=[]
 for part,bounds in values.items():
  for bound,v in bounds.items():
   e=v['vs_base'];rows.append([v['n'],'不利' if bound.endswith('lower') else '有利',v['base_correct'],'/'.join(map(str,v['per_seed_correct'])),f"{e['delta_pp']:+.3f}",ci(e['question_ci95']),ci(e['seed_question_ci95'])])
 text+='### '+label+'\n\n'+table(['题数','边界','Baseline','五seed正确','平均Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n\n'
text+='''所有上述区间均跨零。3500题上，统一方框约定的两种分析中seed44在最有利情形仍仅持平。评分缺陷与表示歧义存在，但这项局部审查没有恢复可靠提升；它不能替代冻结的raw/strict成功标准，也不能证明未审查的评分全部正确。

## 可复核产物

- [全部47条完整题目和输出](audits/math_scoring_disagreement_review_cases.jsonl)
- [逐条数学判断与歧义](audits/math_scoring_disagreement_review_decisions.json)
- [47条敏感性及来源哈希](results/math_manual_disagreement_sensitivity.json)
- [方框算子题全部21个输出复核](audits/math4060_operator_gold_review.json)
- [两种统一解释的敏感性](results/math4060_followup_sensitivity.json)
- [原判断与修订记录](audits/math_operator_semantics_revision.json)

这份文档只覆盖已经完成的评分审计；完整研究的OOD、原生推理与全部干预证据以最终REPORT为准。
'''
path=S/'SCORING_AUDIT.md';path.write_text(text)
write(S/'audits/scoring_audit_report_sources.json',{'time':time.time(),'script_sha256':sha(__file__),'report_sha256':sha(path),'source_sha256':sources,'scope':'Completed partial scoring audit only; does not assert ongoing GPU study complete.'})
print(path)
