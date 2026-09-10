"""Assemble final measured evidence only after all analyses and warning review exist."""
from common import *
from datetime import datetime,timezone
import re,csv

def get(path):return json.loads((S/path).read_text())
def ci(v):return '['+', '.join(f'{x:+.2f}' for x in v)+']'
def yes(v):return '通过' if v else '未通过'
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def counts(f):return '/'.join(str(f['per_seed'][str(seed)]['correct']) for seed in seeds)
plan=get('results/confirmation_plan.json');ph=sha(S/'results/confirmation_plan.json');seeds=plan['seeds'];primary=plan['primary_family']
raw=get('results/confirmation_analysis.json');strict=get('results/confirmation_analysis_strict.json');native=get('audits/final_native_comparison.json');native_strict=get('audits/final_native_comparison_strict.json')
for r in [raw,strict,native,native_strict]:assert r['plan_sha256']==ph
collection=get('audits/final_collection_status.json');assert collection['all_complete'] and collection['plan_sha256']==ph
review=get('audits/final_scoring_warning_review.json');assert review['all_reviewed'] and review['plan_sha256']==ph
assert get('audits/final_math_overlap_replay.json')['passed'] and get('audits/final_native_baseline_replay.json')['passed']
resource=get('audits/resource_scope_final.json');assert resource['passed']
if (S/'audits/balanced_ood_execution_runtime_amendment.json').exists():
 reconciliation=get('audits/final_campaign_execution_reconciliation.json')
 assert reconciliation['passed'] and reconciliation['main_execution_complete'] and reconciliation['plan_sha256']==ph
 for path,digest in reconciliation['source_sha256'].items():assert sha(S/path)==digest,path
if (S/'audits/expanded_unaveraged_supplement_plan.json').exists():
 supplemental_final=get('audits/expanded_supplement_final_review.json');assert supplemental_final['all_reviewed'] and supplemental_final['gpu_work_terminal']
if (S/'audits/native_target_control_plan.json').exists():
 native_target_review=get('audits/native_target_control_final_review.json');assert native_target_review['all_reviewed'] and native_target_review['gpu_work_terminal']
ood_review=get('audits/ood_scoring_disagreement_review.json');assert ood_review['all_reviewed'] and ood_review['n_cases']==74
ood_sensitivity=get('results/ood_manual_scoring_sensitivity.json');assert ood_sensitivity['all_corrected_raw_strict_vectors_equal']
for path,digest in ood_sensitivity['source_sha256'].items():assert sha(S/path)==digest,path
inv=get('results/experiment_inventory.json');decision=get('audits/final_selection_decision.json')
if (S/'audits/initial_learning_rate_screen_catalog.json').exists():
 initial_screen=get('audits/initial_learning_rate_screen_catalog.json')
 for path,digest in initial_screen['source_sha256'].items():assert sha(S/path)==digest,path
labels={'teacher7_avg1234':'旧7B目标，epoch1–4平均','teacher7_expanded_avg1234':'扩容7B目标，epoch1–4平均','teacher7_epoch4':'旧7B目标，未平均epoch4'}
passed=all(raw['criteria'].values()) and all(strict['criteria'].values())
main=raw['datasets']['math_reused5000'];mr=main['families'][primary]['vs_base'];oraw=raw['ood_macro'][primary]
completed=[r for r in inv['training'] if r['status']=='complete'];pending=[r for r in inv['training'] if r['status']=='pending'];failed=[r for r in inv['training'] if r['status']=='failed']
start=datetime(2026,9,9,15,16,40,tzinfo=timezone.utc).timestamp();now=time.time();hours=(now-start)/3600
lead=('固定主配方在本轮预设的完整MATH与数学OOD聚合标准下通过五seed验证，主评分及严格评分均通过。证据支持的是这些已测模型和任务在固定vLLM主协议下的收益，不是任意seed、任意任务都提升。' if passed else '本轮已用受控目标替换识别出原设置的监督构造问题，但固定主配方没有同时通过全部预设的五seed MATH/OOD标准。因此不能宣布已经修复为稳定的一般性提升，也不能把有限失败改写成“LoRA不能SFT”。')
parts=[f'''# Qwen2.5-1.5B-Instruct 的 LoRA SFT：失败原因与五seed验证

**{lead}**

研究开始：2026-09-09 15:16:40 UTC。报告生成：{datetime.fromtimestamp(now,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}，已用{hours:.2f}小时，硬截止2026-09-10 15:16:40 UTC。初期上限为两张GPU，使用GPU1/3；用户随后明确允许三张卡，于2026-09-10 06:27:57 UTC登记[执行修订](audits/gpu2_additional_execution_amendment.json)后增加GPU2，最多同时三张，原时间截止不变。逐题输出、未合并adapter、配置、数据哈希和失败记录均保留。

## 对问题的直接回答

本轮证据支持**原始参考解答监督存在可由受控实验定位的问题**：它与当前优化协议的组合造成了可重复的自由生成损害。但你追问的是此前修复之后为何仍没有稳定超过baseline。对这一更进一步的问题，目标替换的恢复证据只解释了原方案的大幅退化；它并未单独证明残余泛化瓶颈的唯一原因，也未给出已验证充分的修复配方。下文分别报告已定位的干预效应与最终多seed泛化结果。

最明确的原因不是“LoRA参数没有更新”，而是**当前原始参考解答监督与训练配置的组合，优化了参考文本的模仿，却大幅损害自由生成答题；替换监督可以修复大部分损害，但残余泛化收益必须另行验证**。两个同题控制的生成目标减raw为+24.8与+27.8个百分点，严格评分更大，并有上轮不同题集三seed同方向证据。格式/截断输出的乐观上界仍不足以弥补差距。

这里能精确识别的是目标构造这一整体干预。长度、推导内容、表达和tokenization同时改变，不能声称已隔离出唯一的“文风”“NLL”或某层参数机制。学习确实发生：已见题自由生成明显改善；开发新题上，救回原错误题与丢掉原正确题相抵。增加rank、更大教师、DFT、KL、最低NLL选择、多解答及扩容等具体干预的证据分别列出，未成功分支没有丢弃。

正式冻结主候选为**{labels[primary]}**。完整MATH5000上五seed为**{counts(main['families'][primary])}**，baseline **{main['base_correct']}**；均值差**{mr['delta_pp']:+.2f} pp**，题目区间{ci(mr['question_ci95'])}，seed/题目区间{ci(mr['seed_question_ci95'])}。预留数学OOD等权宏平均差**{oraw['delta_pp']:+.2f} pp**，两种区间{ci(oraw['question_ci95'])}、{ci(oraw['seed_question_ci95'])}。以下给出所有seed、任务、对照和严格敏感性。

完整MATH的实际变化并非完全不学习：五个seed平均救回baseline答错的413.8题，同时丢掉baseline原本答对的373.6题，净增仅40.2/5000。也就是说，提升与损失大部分相抵；同一模型换到Olympiad及SVAMP时，五个seed的主评分均低于baseline。这个逐题分解精确描述了当前没有稳定净收益的表现，但尚不能单独把相抵归因于容量、目标token权重或某个优化机制。

区间跨零表示当前证据不足以确定正向收益，不能据此断言真实效应恰好为零。有限配方、任务和seed的失败也不构成LoRA普遍不能SFT的证明。

评分器的敏感性在OOD上尤其大：主方案严格宏平均为+4.23 pp，而主评分为−1.91 pp。逐条复核确认，SVAMP baseline的56个raw/strict分歧中，54个是明确的正确无框答案，另2个有题面时序歧义；五个主候选的SVAMP分数在两种规则下完全不变。因此不能将严格评分的正增益解释成一般推理能力提升。完整74条分歧复核后，即使另行补回3份正确百分数答案、并给4条歧义最有利于SFT的赋值，主方案OOD宏平均仍为−1.69 pp，五个seed仍全部低于baseline；不利边界为−1.94 pp。两种seed/题目区间仍跨零，因此这是未显示正向泛化的证据，不是确证总体负效应。逐题向量和重新抽样的边界分析见后文；冻结的原始分数均保留。

## 研究设计与可外推范围

学生与baseline都是已经经过指令后训练的Qwen2.5-1.5B-Instruct，固定snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。本轮70个未续训运行的零B及同seed/同结构初始权重核对通过，续训分支单独标记，见[初始化审计](audits/initialization_all_current.json)。旧研究的mask、causal shift、真实梯度累积、保存加载及attention算术证据直接复用，本轮只审计新增路径，避免重做已完成的实验。[前轮报告](../sft_diagnosis_20260908/causal_followup/REPORT.md)。

报告中的逐题数学复核由本助手执行，保留原文、计算或反例及文件哈希；不是独立人类盲评。事后追加的审查明确标注结果已知，正确最终答案不自动代表推导有效。未解决的歧义保留并做边界分析，不通过选择有利裁定更改成功标准。

开发500题来自官方MATH train的历史留出，**不是官方MATH-500**。本轮用于选模的1500题和完整test5000均有历史复用；不能称为独立未触及测试。预先固定的补充分析排除本轮选模1500题，保留3500题，它仍有历史复用。新增OOD878题在配方冻结前没有模型输出：Minerva272、Olympiad266、SVAMP300按任务准确率等权平均，AMC23的40题只作补充。它们衡量数学及定量推理迁移，不代表非数学通用能力，也不是每个数据集全部官方协议的复现。下载版本、文件哈希及原始来源见[固定下载记录](audits/ood_downloads.json)。

两候选的开发选择规则在四个扩容seed训练前固定：扩容五seed只有在原始及严格评分下都全部高于开发baseline、且均值都高于旧平均时，才取代旧候选。三个五seed家族无论被选与否都完成最终评测；权重、数据、端点、脚本及协议在任何本轮完整5000/OOD输出前冻结。[规则](audits/final_selection_rule.json)、[实际决定](audits/final_selection_decision.json)、[冻结方案](results/confirmation_plan.json)。

训练主要使用QKVO LoRA r16/alpha16/dropout0，冻结BF16基座、FP32 A/B与AdamW状态、SDPA math，microbatch4、有效batch16、completion token-mean CE、clip1、weight decay0.01、64条样本warmup、8轮cosine horizon并固定在epoch4停止。平均模型通过拼接因子实现四个有效更新BA的数学平均，推理rank64；不是分别平均A/B，也不是预测投票。数学等式和BF16运行时数值等价须区别。[详细方法](METHODS.md)、[复现](REPRODUCE.md)。

主推理协议固定BF16、vLLM TRITON、未合并LoRA、batch-invariant、greedy、2048新token、4096总context、每批最多500题。所有对照共享baseline协议，不混用历史FP32的304/500与当前BF16的299/500。MATH等任务的主评分复用完整文本解析，严格最后完整boxed评分作敏感性；Minerva的191道数值题在两种评分中都使用下文的前置数值修复；全部parser warning逐条复核而不挑有利评分。

## 固定五seed的完整结果

下列计数依次为seed43/44/45/46/47。区间以题目为配对单位；跨seed区间共享题目抽样，避免把同一题的五次回答当成五道独立题。主判断要求完整MATH与OOD宏平均各自五seed全正、题目及seed/题目95%区间下界都严格大于零。有限五seed不是所有可能seed的保证；OOD宏平均为正也不保证每个benchmark都提升，因此各任务与各seed逐项列出。宏平均的题目区间在三个固定任务内重采样题目；seed/题目区间还重采样这五个训练seed。两者都不包含从所有数学任务中抽选benchmark的不确定性。非主家族与对照的区间保持探索性，不能替代主标准。
''']
for title,result in [('主评分（Minerva数值已按前置方案修正）',raw),('严格最后答案框评分',strict)]:
 rows=[]
 for dataset,res in result['datasets'].items():
  for family,fr in res['families'].items():
   v=fr['vs_base'];rows.append([dataset,labels[family]+('（主）' if family==primary else ''),f"{res['base_correct']}/{res['n']}",counts(fr),f"{v['delta_pp']:+.2f}",ci(v['question_ci95']),ci(v['seed_question_ci95'])])
 parts.append('### '+title+'\n\n'+mdtable(['集合','配方','Baseline','五seed正确数','均值Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n')
 rows=[]
 for family,v in result['ood_macro'].items():rows.append([labels[family],'/'.join(f'{x:+.2f}' for x in v['per_seed_delta_pp']),f"{v['delta_pp']:+.2f}",ci(v['question_ci95']),ci(v['seed_question_ci95']),yes(v['every_seed_positive'])])
 parts.append(mdtable(['数学OOD宏平均','五seedΔ pp','均值Δ pp','题目95%CI','seed/题目95%CI','每seed为正'],rows)+'\n')
 parts.append('主标准：'+ '；'.join(f'{k}：{yes(v)}' for k,v in result['criteria'].items())+'。\n')
parts.append('![主候选五seed及配对区间](figures/confirmation_primary.png)\n\n主/严格逐题统计与所有控制差见[主分析](results/confirmation_analysis.json)、[严格分析](results/confirmation_analysis_strict.json)。图片的CSV、SVG、PDF及源哈希均在figures目录。\n')
parts.append('### 主候选与两个固定对照的直接配对差\n\n这些是次要比较，未校正全部探索历史；相对baseline为正不等于显著优于另一个SFT配方。\n')
for mode,result in [('主评分',raw),('严格评分',strict)]:
 rows=[]
 for dataset in ['math_reused5000']:
  for comparison,v in result['datasets'][dataset]['comparisons'].items():
   rows.append([dataset,labels[primary]+' 减 '+labels[comparison.split('_vs_',1)[1]],f"{v['delta_pp']:+.2f}",ci(v['question_ci95']),ci(v['seed_question_ci95'])])
 for comparison,v in result['ood_macro_controls'].items():
  rows.append(['数学OOD宏平均',labels[primary]+' 减 '+labels[comparison.split('_vs_',1)[1]],f"{v['delta_pp']:+.2f}",ci(v['question_ci95']),ci(v['seed_question_ci95'])])
 parts.append(mode+'。\n\n'+mdtable(['集合','主候选减对照','均值Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n')
parts.append('### Minerva数值评分的前置修复\n\n在任何OOD模型输出前发现，58条科学计数法参考如4.5e33被通用LaTeX解析器当作含欧拉常数的表达式；默认比较还会接受10^-19与2×10^-19。191道数值题现按安全科学计数法解析、有限实数、相对容差1e-4及绝对容差0比较，并要求最后完整boxed；81道符号题沿用原始/严格两种相应规则。所有数据、生成代码和旧分数保留，新的分析向量单独保存。191道数值参考的等价表示/错误值控制及15组表示、单位及下溢测试通过。\n\n相对1e-4采用[Qwen官方评测的数值比较约定](https://github.com/QwenLM/Qwen2.5-Math/blob/615096eceb14e653ba791d862a21b326a1d26e12/evaluation/grader.py)，不是完整复现原[Minerva附录F.5](https://arxiv.org/html/2206.14858v2#A6.SS5)的提取、单位与近零分支。另预置1%和5%容差诊断及全部旧评分敏感性，它们不被用于事后放宽成功标准。这是本轮新增OOD准备时发现的评分问题，不能解释研究开始前的MATH退化。[修复方案](audits/minerva_numeric_scoring_amendment.json)、[数值契约](audits/minerva_numeric_contract.json)、[修正评分](audits/scoring_minerva_numeric_corrected.json)。\n')
for mode,result in [('主评分',raw),('严格评分',strict)]:
 rows=[]
 for tolerance,families_sens in result['minerva_numeric_sensitivity'].items():
  v=families_sens[primary];m=v['ood_macro'];rows.append([tolerance,v['minerva_base_correct'],'/'.join(map(str,v['minerva_per_seed_correct'])),f"{m['delta_pp']:+.2f}",ci(m['question_ci95']),ci(m['seed_question_ci95']),yes(m['every_seed_positive'])])
 parts.append(mode+'，所选主配方的评分敏感性。\n\n'+mdtable(['Minerva评分','Minerva baseline','Minerva五seed','OOD宏均值Δ pp','题目95%CI','seed/题目95%CI','每seed为正'],rows)+'\n')
parts.append('### 排除当前选模1500题后的3500题与重叠敏感性\n')
for title,r in [('原始',raw),('严格',strict)]:
 h=r['validation_excluded_sensitivity'];rows=[]
 for family,v in h['families'].items():
  x=v['vs_base'];rows.append([labels[family],'/'.join(str(v['per_seed_correct'][str(seed)]) for seed in seeds),f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95']),yes(x['every_seed_positive'])])
 parts.append(f"{title}评分，baseline {h['base_correct']}/3500。\n\n"+mdtable(['配方','五seed正确数','均值Δ pp','题目95%CI','seed/题目95%CI','每seed为正'],rows)+'\n')
parts.append('该3500题集合在本轮完整5000输出前固定。另报告保守移除三个字符相似候选后的4997题敏感性；三条不是三条已确认泄漏，部分是负号归一化的假匹配。语义去重也不能证明无预训练污染。完整5000中所有可对照的历史1500输出已逐题核查全文、评分、长度与停止原因，见[重放审计](audits/final_math_overlap_replay.json)。\n')
for mode,result in [('主评分',raw),('严格评分',strict)]:
 rows=[]
 for dataset,v in result['sensitivity'].items():
  x=v['families'][primary]
  rows.append([dataset,v['n'],v['base_correct'],f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95']),yes(x['every_seed_positive'])])
 parts.append(mode+'，保守重叠排除后的所选主候选。\n\n'+mdtable(['集合','题数','Baseline','均值Δ pp','题目95%CI','seed/题目95%CI','每seed为正'],rows)+'\n')
parts.append('开发500题的435个可识别实数参考以及完整MATH的4387个实数参考另做了“非零值乘2、零值加1”的错误答案测试。开发集未见误接受，完整MATH仅index2854会把2/2004!误判为正确1/2004!。该题不在本轮1500选模集合中；保留完整5000原主分数，并在任何最终输出前固定去除此题的4999题及再合并三个保守重叠候选的4996题敏感性。这只是具体负例测试，不是通用评分器正确性的证明。[负例审计](audits/math_numeric_negative_controls.json)、[前置敏感性](audits/math_numeric_grader_sensitivity_plan.json)。\n')
for label,r in [('主',raw),('严格',strict)]:
 rows=[]
 for name,v in r['math_numeric_grader_sensitivity'].items():
  x=v['families'][primary];rows.append([name,v['n'],v['base_correct'],f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95']),yes(x['every_seed_positive'])])
 parts.append(label+'评分、所选主候选。\n\n'+mdtable(['排除方案','题数','Baseline','均值Δ pp','题目95%CI','seed/题目95%CI','每seed为正'],rows)+'\n')
parts.append('### 原生HF/PEFT后端复核\n\n所选主候选的全部五seed与baseline在同500道复用开发题上重新自由生成。比较各自后端内部的adapter−base；HF与vLLM的batch、dtype细节及后端共同变化，不能把差异归于其中一项。\n')
for title,r in [('原始',native),('严格',native_strict)]:
 rows=[[str(x['seed']),x['native_correct'],f"{x['native_delta_pp']:+.2f}",x['vllm_correct'],f"{x['vllm_delta_pp']:+.2f}"] for x in r['rows']]
 parts.append(f"{title}评分：HF baseline {r['native_base_correct']}/500，vLLM baseline {r['vllm_base_correct']}/500。\n\n"+mdtable(['seed','HF正确数','HF Δ pp','vLLM正确数','vLLM Δ pp'],rows)+'\n')
 n=r['native_vs_native_base'];d=r['native_minus_vllm_treatment_effect'];parts.append(f"HF平均效应{n['mean_delta_pp']:+.2f} pp，题目区间{ci(n['question_ci95_pp'])}、seed/题目区间{ci(n['seed_question_ci95_pp'])}；全部seed为正：{yes(n['all_seeds_positive'])}。HF减vLLM的处理效应差{d['mean_delta_pp']:+.2f} pp，题目区间{ci(d['question_ci95_pp'])}。跨零不能证明后端等价。\n")
parts.append('原生baseline前128题与先前原生记录逐项重放，所有adapter加载后的FP32 A/B逐位核对。[原生分析](audits/final_native_comparison.json)、[严格原生分析](audits/final_native_comparison_strict.json)、[baseline重放](audits/final_native_baseline_replay.json)。\n')
causal=(S/'CAUSAL_FINDINGS.md').read_text();causal=causal[causal.index('## 1.'):]
causal=re.sub(r'^## ', '### ',causal,flags=re.M)
causal=causal.replace('最终候选还将用HF/PEFT原生推理复查全部五seed及baseline。','最终主候选已用HF/PEFT原生推理复查全部五seed及baseline，具体结果见前文。')
causal=causal.replace('最终证据将按固定五seed、完整MATH、等权数学OOD、严格评分及原生后端分别报告，主标准不在看到结果后放宽。','最终证据已按固定五seed、完整MATH、等权数学OOD、严格评分及原生后端分别报告，主标准保持不变。')
parts.append('## 为什么原方案退化：受控证据与尚未识别的因素\n\n'+causal)
matched=get('results/matched_generator_control_analysis.json');parts.append('### 严格相同问题的生成器来源对照\n')
for scoring,r in matched['results'].items():
 rows=[[p['hypothesis'],p['treatment_correct'],p['control_correct'],f"{p['delta_pp']:+.2f}",ci(p['ci95_pp'])] for p in r['pairs']]
 parts.append(scoring+'评分。\n\n'+mdtable(['处理减对照','处理正确数','对照正确数','Δ pp','配对95%CI'],rows)+'\n')
parts.append('学生/7B共有1429题；32B/7B共有1465题。全部问题顺序及来源行逐项核对；32B/7B目标token734689/675695，相差约8.7%，不是长度完全匹配。教师参数规模、过程、内容及表达共同变化，因而是来源整体干预。[数据契约](audits/matched_generator_data_contract.json)、[配对结果](results/matched_generator_control_analysis.json)。\n')
parts.append('''## 文献提供的约束与本地实验证据的区别

[Qwen2.5技术报告](https://arxiv.org/abs/2412.15115)描述规模化指令后训练，包括数学监督；本地Instruct不是未学过数学的Base，官方数字与本地协议不同，也不能把分数接近当能力上限。[STaR](https://arxiv.org/abs/2203.14465)、[RFT](https://arxiv.org/abs/2308.01825)、[Mind the Gap](https://arxiv.org/abs/2509.15157)启发正确轨迹与覆盖干预，但本轮不是这些完整算法的大规模复现。

[S3FT](https://arxiv.org/abs/2502.08130)已有其他模型任务上的LoRA监督训练正结果。[DFT附录A.6/Table9](https://arxiv.org/html/2508.05629v3)报告Qwen2.5-Math-1.5B、LoRA r8/alpha16的Math500为Base31.66、SFT41.47、DFT64.85；它使用数学Base、100k监督、16次随机解码及4096token，不能替代本地Instruct五seed、greedy证据。这些结果反对“任何LoRA都不能SFT”的概括，不能保证本任务成功。普通CE对logits的梯度为p减one-hot，不能仅凭某种逆概率重写认定本地梯度爆炸。

[ASFT](https://arxiv.org/html/2509.23753v3)使用DFT与forward KL；其医疗问答较优系数不是本地数学已知最优值。本轮在任何对应DFT结果前固定CE/DFT×KL0/.1/1网格，完整报告不成功的比较。[SmartAD](https://aclanthology.org/2026.findings-acl.1349.pdf)更接近同一1.5B-Instruct与2000 MATH问题，但Table2的数学CoT Prompt48.2高于CoT Distill44.2和SmartAD44.4；不能把多任务平均或工具能力的收益说成纯数学SFT已超过baseline。

[OpenMathInstruct-2](https://arxiv.org/abs/2410.01560)主要研究Llama-8B Base且其Table3过滤未改善，不能作为当前小Instruct数据配方的保证。[过程监督研究](https://arxiv.org/abs/2501.07301)也要求区别正确结果与正确推导。[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)、[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)涉及容量、遗忘和优化；loss改善不能直接转换为这里的自由生成准确率。论文、网页摘录与版本记录见[literature](literature/NOTES.md)。

[CFT论文Table6](https://arxiv.org/html/2510.10974v1#A4.T6)另报告Qwen2.5-3B的MATH未微调34.6、LoRA SFT41.9、LoRA CFT49.0；其逐token反事实续写标注与超参搜索提供尚未测试的监督分配假说，未在本轮实现，也没有本地1.5B-Instruct五seed证据。有限候选续写失败不证明所有替代推导都失败。公开同1.5B模型的单次100题提升或提取器分数分歧也不满足本任务标准，具体核查见[补查记录](literature/late_crosscheck/NOTES.md)。

## 运行审计、资源与复现

本轮曾发生一次GPU1交接竞态，导致同一张GPU上的生成和训练短暂并行，仍只使用GPU1/3。正式生成重新串行执行；对应2048条旧部分输出全文、评分、长度完全一致，受影响训练的初始及四epoch全部adapter张量、曝光顺序也与独占重跑逐位一致。正式覆盖分析使用串行重跑，不把这件本轮事故当作原现象原因。

另一次评测启动因PATH未包含现有ninja而在生成前失败，修复路径后baseline与非零adapter各500题完整复放。新CPU冻结守卫也曾把仅调度不同的两个已验证版本误要求为同一源码哈希；修正只涉及审计守卫，保留旧脚本/日志，并核对AST仅队列分支不同以及1000条复放完全一致。训练或推理数值代码没有因此改变。[串行复跑](audits/affected_training_serial_replay_result.json)、[路径恢复](audits/evaluation_path_startup_recovery.json)、[版本桥接](audits/evaluator_version_bridge.json)。
''')
parts.append(f"实际清单记录完成训练{len(completed)}项，未执行排队计划{len(pending)}项，训练失败{len(failed)}项，派生模型{len(inv['derived'])}项，评测汇总{len(inv['evaluations'])}份。计数包含数值重放和共享训练历史的续训标记；checkpoint数、平均模型数与独立seed数不可混同。记录的GPU命令、训练manifest及最终自身进程分配审计通过；whole-device资源日志包含其他用户负载，不能当成本研究独占FLOPs。最终推理期间的[在线快照](audits/shared_load_final_snapshot.json)确认同卡还存在其他用户分配；该快照不能重建全部历史起止，也没有单独识别共享负载对速度或分数的因果效应。[机器可读清单](results/experiment_inventory.json)、[CSV](results/experiment_inventory.csv)、[资源审计](audits/resource_scope_final.json)。\n")
parts.append('最终模型及其路径以[固定manifest](results/confirmation_manifest.json)为准，训练数据和每个文件的哈希见[方案](results/confirmation_plan.json)。直接加载对应adapter目录即可获得该冻结模型；平均adapter是有效更新平均后的单模型。评测命令、依赖环境、全部派生与原生步骤见[复现说明](REPRODUCE.md)。本报告的范围、未成功分支与数据复用限制均适用于模型使用；不能将其包装为普遍的LoRA不可能性证明。\n')
if (S/'audits/balanced_ood_execution_runtime_amendment.json').exists():
 parts.append('在任何实际OOD输出前，另作[队列执行修订](audits/balanced_ood_execution_runtime_amendment.json)：GPU2和GPU3各自正在运行的MATH子进程自然完成后，共同处理冻结OOD任务，原16模型优先，五个补充模型随后。GPU1原生复核持续运行。只暂停并退役CPU调度父进程，没有中断当时的GPU推理；新派发工作截止15:00 UTC，研究硬截止仍为15:16:40。原主调度器的benchmark返回码-15保留，实际完成情况由[执行核对](audits/final_campaign_execution_reconciliation.json)连接原MATH正常退出、替代OOD完成以及原生/收集器正常退出的证据，不将原返回码伪装成零。\n')
if (S/'audits/balanced_ood_zombie_read_amendment.json').exists():
 parts.append('第一次实际MATH交接还发现CPU进程读取器的错误：已正常退出的子进程仍可读取PID出生时间、父进程及退出码，但读取其`/proc/environ`被拒绝，旧读取器错误地返回“进程丢失”。当时没有派发任何OOD任务。保留旧源码和错误状态后，以[单独修订](audits/balanced_ood_zombie_read_amendment.json)让退出进程只通过UID、出生时间、父进程和内核退出码验证，存活父进程仍检查环境；实际退出0和错误出生时间拒绝检查通过。随后继续原冻结队列，未中断GPU1/2推理，也未重跑已完成MATH。这是本轮调度记录读取错误，不是原SFT精度问题。[实际进程检查](audits/balanced_ood_zombie_read_contract.json)。\n')
text='\n'.join(parts)
assert '研究进行中' not in text
(S/'REPORT.md').write_text(text)
sourcepaths=['audits/ood_scoring_disagreement_review.json','results/ood_manual_scoring_sensitivity.json','results/confirmation_plan.json','results/confirmation_analysis.json','results/confirmation_analysis_strict.json','audits/final_native_comparison.json','audits/final_native_comparison_strict.json','audits/final_scoring_warning_review.json','results/experiment_inventory.json','CAUSAL_FINDINGS.md']
if (S/'audits/initial_learning_rate_screen_catalog.json').exists():sourcepaths.append('audits/initial_learning_rate_screen_catalog.json')
if (S/'audits/final_campaign_execution_reconciliation.json').exists():sourcepaths.append('audits/final_campaign_execution_reconciliation.json')
write(S/'audits/final_report_sources.json',dict(time=time.time(),script_sha256=sha(__file__),report_sha256=sha(S/'REPORT.md'),source_sha256={f:sha(S/f) for f in sourcepaths},primary=primary,raw_and_strict_prespecified_criteria_passed=passed))
print(json.dumps(dict(report=str(S/'REPORT.md'),primary=primary,criteria_passed=passed,training_complete=len(completed),hours=hours)))
