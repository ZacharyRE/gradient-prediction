"""Append measured diagnostic tables after terminal manual reviews; no new scoring."""
from common import *

def get(p): return json.loads((S/p).read_text())
def ci(v): return '['+', '.join(f'{x:+.2f}' for x in v)+']'
def seq(v): return '/'.join(map(str,v))
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])

path=S/'REPORT.md'; before=sha(path); report=path.read_text()
assert '## 补充诊断：后端、验证迁移与错误实例' not in report
plan=get('results/confirmation_plan.json'); primary=plan['primary_family']
labels={'teacher7_avg1234':'旧7B，平均','teacher7_epoch4':'旧7B，未平均',
        'teacher7_expanded_avg1234':'扩容7B，平均','teacher7_expanded_epoch4':'扩容7B，未平均'}
sources=[]
def source(p): sources.append(p); return get(p)
parts=['## 补充诊断：后端、验证迁移与错误实例\n']
review=source('audits/native_target_control_final_review.json')
assert review['all_reviewed'] and review['gpu_work_terminal']
status=source('audits/native_target_control_status.json')
if status['all_complete']:
    r=source('results/native_target_control_analysis.json')
    parts.append('### 原始目标退化是否只出现在vLLM\n\n在固定seed43的同128题上，补测两个原始目标模型和对应7B生成目标模型，复用已核对的原生baseline与学生生成目标输出。用户允许三张卡后，这组额外控制提前到GPU2执行；既有baseline与学生模型先在GPU2各重放128题，并要求与GPU1记录逐项一致，见[跨卡复放](audits/gpu2_native_protocol_replay.json)。每个后端内部比较相同模型；这不是新五seed验证。以下所有区间仅表示题目抽样不确定性，单seed不能估计训练seed的总体变化。\n')
    for mode,v in r['results'].items():
        rows=[]
        for backend,b in v['backends'].items():
            for p in b['pairs']:
                e=p['generated_minus_raw']
                rows.append([backend,'学生生成' if p['generated'].startswith('sample_') else '7B生成',
                             b['base_correct'],p['raw_correct'],p['generated_correct'],
                             f"{e['delta_pp']:+.2f}",ci(e['question_ci95'])])
        parts.append(mode+'评分。\n\n'+table(['后端','目标来源','Baseline/128','原始目标/128','生成目标/128','生成减原始 pp','题目95%CI'],rows)+'\n')
        rows=[]
        for p in v['backend_target_interactions']:
            e=p['native_minus_vllm_target_effect']
            rows.append(['学生' if p['generated'].startswith('sample_') else '7B',f"{e['delta_pp']:+.2f}",ci(e['question_ci95'])])
        parts.append(table(['来源','HF减vLLM的目标干预效应差 pp','题目95%CI'],rows)+'\n')
    parts.append('该对照可以检验原始目标损害是否跨这两个已测推理配置存在；不能把HF/vLLM差异归于单独一个kernel、adapter dtype或batch配置。完整模型身份、逐题配对与warning记录见[原生目标控制](results/native_target_control_analysis.json)。\n')
    if review.get('report_markdown'):
        parts.append(review['report_markdown']+'\n')
    a=r['reference_ce_alignment_same_native64']
    rows=[['Baseline',f"{a['base_reference_ce']:.6f}",a['base_native_correct'],'—',a['base_vllm_correct'],'—']]
    for x in a['rows']:
        label=('学生' if x['model'].startswith('sample_') else '7B')+('原始目标' if x['kind']=='raw' else '生成目标')
        rows.append([label,f"{x['reference_ce']:.6f}",x['native_correct_same64'],x['native_strict_correct_same64'],x['vllm_correct_same64'],x['vllm_strict_correct_same64']])
    parts.append('### 同一HF框架、同64题上的参考CE与自由生成\n\n参考CE与原生自由生成使用相同64题及checkpoint。teacher forcing与自回归、batch/cache仍不同，因此它检验loss代理与实际生成收益的对应关系，不是两个数值执行路径完全等价的证明。\n\n'+table(['模型','原始参考CE','HF正确/64','HF严格/64','vLLM正确/64','vLLM严格/64'],rows)+'\n')
else:
    parts.append('原生原始目标补充控制未全部完成；保留实际终止状态，不能据此排除目标与推理后端的交互解释。见[执行状态](audits/native_target_control_status.json)。\n')

if (S/'audits/training_gradient_log_plan.json').exists():
    gradient_plan=source('audits/training_gradient_log_plan.json')
    gradients=source('results/training_gradient_log_analysis.json')
    assert gradients['plan_sha256']==sha(S/'audits/training_gradient_log_plan.json')
    assert gradients['script_sha256']==sha(S/'scripts/analyze_training_gradient_logs.py')
    for p,digest in gradients['source_sha256'].items(): assert sha(S/p)==digest,p
    rows=[]
    for name,v in gradients['results'].items():
        if name.startswith('teacher7_expansion_'): label='扩容7B，seed'+name.rsplit('_s',1)[1]
        elif name.startswith('teacher_all_'): label='旧7B生成，seed'+name.rsplit('_s',1)[1]
        elif name.startswith('teacher_matchedraw_'): label='7B同题raw，seed43'
        elif name.startswith('sample_matchedraw_'): label='学生同题raw，seed43'
        else: label='学生生成，seed43'
        a=v['all_steps']
        rows.append([label,v['endpoint_epoch'],a['steps'],f"{a['median']:.4f}",f"{a['p95']:.4f}",f"{a['max']:.4f}",a['epsilon_clipping_proxy_count']])
    parts.append('### 梯度是否被clip1限制\n\n在结果已知后，按[日志诊断声明](audits/training_gradient_log_plan.json)检查两个同题raw/生成目标对照的固定端点，以及旧7B和扩容7B全部五seed的训练日志。记录值来自`clip_grad_norm_`返回的裁剪前总范数，覆盖端点前每一步，包括最初LR0的一步；本地PyTorch使用`1/(norm+1e-6)`并将系数上限设为1。13个实际运行的7920步均为有限、非零范数，最大0.7768，安全低于触发阈值，因此这些步骤没有被clip1缩小。这排除了“这批模型因为频繁触发clip1而学不动”的具体解释。它不证明梯度方向正确，也不将梯度范数当作AdamW实际权重更新量。\n\n'+table(['训练来源','截至epoch','步数','范数中位','P95','最大','触发裁剪步数'],rows)+'\n\n逐epoch值、全部history/exposure/source哈希见[日志分析](results/training_gradient_log_analysis.json)及[CSV](results/training_gradient_log_analysis.csv)。没有新增训练或据此更换主模型。\n')

if (S/'audits/math_scoring_disagreement_review_plan.json').exists():
    manual=source('results/math_manual_disagreement_sensitivity.json')
    follow=source('results/math4060_followup_sensitivity.json')
    decisions=source('audits/math_scoring_disagreement_review_decisions.json')
    operator=source('audits/math4060_operator_gold_review.json')
    assert manual['all47_reviewed'] and decisions['all_reviewed'] and operator['all_reviewed']
    assert manual['script_sha256']==sha(S/'scripts/analyze_math_manual_disagreements.py')
    assert follow['script_sha256']==sha(S/'scripts/analyze_math4060_followup.py')
    for result in [manual,follow]:
        for p,digest in result['source_sha256'].items(): assert sha(S/p)==digest,p
    rows=[]
    for n,v in manual['counts'].items():
        rows.append(['Baseline' if n=='base' else '主方案seed'+n.split('_s')[-1].split('_')[0],v['disagreements'],v['correct'],v['incorrect'],v['unresolved']])
    parts.append('### 主方案全部评分分歧的逐题复核\n\n在完整MATH的raw/strict分数已知后，额外声明并读完baseline与主方案五seed的全部47条评分分歧，覆盖35道题；不是盲评。最终答案正确与推导有效分别处理：例如某些正确无框答案仍含错误推导，不能因此计为有效证明。4条输出因未完成或自相矛盾保留歧义，另4条因题目方框算子与答案标记冲突保留歧义，共8条；不选择有利解释。其他raw/strict一致的判断并未得到全面数学认证。\n\n'+table(['模型','分歧数','答案正确','答案错误','保留歧义'],rows)+'\n')
    rows=[]
    analyses=[('仅47分歧，独立歧义边界',manual['results'])]+[('方框统一为答案标记' if interpretation=='answer_markup' else '方框统一为约数算子',values) for interpretation,values in follow['results_by_interpretation'].items()]
    for label,values in analyses:
        for partition,bounds in values.items():
            for bound,v in bounds.items():
                e=v['vs_base'];rows.append([label,partition,'不利边界' if bound.endswith('lower') else '有利边界',v['base_correct'],seq(v['per_seed_correct']),f"{e['delta_pp']:+.3f}",ci(e['question_ci95']),ci(e['seed_question_ci95'])])
    parts.append(table(['敏感性','题目','歧义赋值','Baseline','五seed正确','平均Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n\n边界对baseline歧义与SFT歧义作相反赋值，以包住平均效应；各模型共享同一个baseline和配对题目。每个边界均用冻结的`compare()`重新计算bootstrap，不能把单题改判导致的均值变化直接平移整个区间。这些是事后、局部人工复核的敏感性分析，不能替代原先冻结的raw/strict成功标准。\n\n额外追查index4060的全部21个实际输出发现：题目将方框定义为约数个数算子，目标值为d(d(11)×d(20))=6；但通常推理提示又将方框用作最终答案标记。7份最终boxed12在答案标记解释下是错误12，在字面约数算子解释下却等于正确6，即使其过程错误地只计算了内层乘积。其他14份boxed8/60/66在两种解释下都错。最初将21份一概判错的审查过强，已保留旧记录并作[语义修订](audits/math_operator_semantics_revision.json)。追加分析对baseline及全部模型一致采用每种约定，再对其余4条输出歧义计算边界；不能给baseline与SFT分别挑有利约定。原始数据、评分及主成功标准全部保留。\n\n[逐条判断与独立数学检查](audits/math_scoring_disagreement_review_decisions.json)、[47条敏感性](results/math_manual_disagreement_sensitivity.json)、[方框运算题21模型全文复核](audits/math4060_operator_gold_review.json)、[追加敏感性](results/math4060_followup_sensitivity.json)。\n')

review=source('audits/ood_scoring_disagreement_review.json');assert review['all_reviewed']
ood_manual=source('results/ood_manual_scoring_sensitivity.json');assert ood_manual['all_corrected_raw_strict_vectors_equal']
for p,digest in ood_manual['source_sha256'].items():assert sha(S/p)==digest,p
rows=[]
for unit_mode,bounds in ood_manual['results'].items():
    for bound,families in bounds.items():
        for family,v in families.items():
            e=v['ood_macro'];rows.append(['仅判分分歧' if unit_mode=='disagreement_only' else '另含已确认百分数答案','不利边界' if bound=='effect_lower' else '有利边界',labels[family],f"{e['delta_pp']:+.3f}",ci(e['question_ci95']),ci(e['seed_question_ci95']),seq([round(x,3) for x in e['per_seed_delta_pp']])])
parts.append('### OOD评分分歧与格式造成的表面正增益\n\n'+review['report_markdown']+'\n\n'+table(['人工敏感性','歧义赋值','配方','OOD宏均值Δ pp','题目95%CI','seed/题目95%CI','五seedΔ pp'],rows)+'\n\n全部74条实际分歧及另行复核的3个正确百分数输出只进入事后敏感性；没有改动冻结分数或成功标准。两个边界覆盖四个已知歧义，不涵盖所有尚未审查的一致判分、参考质量或任务抽样不确定性。修正后raw/strict逐题向量完全一致已程序核对。区间从逐题向量重新抽样计算，未按点差平移。\n\n[逐例完整审查](audits/ood_scoring_disagreement_review.json)、[全模型覆盖清单](audits/ood_scoring_disagreement_inventory.json)、[分析声明](audits/ood_manual_sensitivity_plan.json)、[重新计算结果](results/ood_manual_scoring_sensitivity.json)。\n')

r=source('results/math_validation_transfer.json')
parts.append('### 当前选模1500题与其余3500题\n\n两部分题目在最终推理前已固定；下面的直接差分析是在观察到前两个主模型的5000题结果后增加的描述性诊断，不能当作预注册主检验。该差异可能涉及题目构成、抽样及适应性选模，不能仅凭分数差证明哪一种造成了差异。各已完成家族的全部五seed一起报告，未挑选收益较大的seed。\n')
for mode,families in r['results'].items():
    rows=[];contrasts=[]
    for family,v in families.items():
        for partition,p in v['partitions'].items():
            e=p['vs_base'];rows.append([labels[family],p['n'],p['base_correct'],seq(p['per_seed_correct']),
                seq(p['per_seed_rescues']),seq(p['per_seed_losses']),f"{e['delta_pp']:+.2f}",ci(e['seed_question_ci95'])])
        e=v['selected1500_minus_remaining3500'];contrasts.append([labels[family],f"{e['delta_pp']:+.2f}",ci(e['question_ci95']),ci(e['seed_question_ci95'])])
    parts.append(mode+'评分。\n\n'+table(['配方','题数','Baseline','五seed正确','救回原错误','丢失原正确','均值Δ pp','seed/题目95%CI'],rows)+'\n\n'+table(['配方','1500减3500效应差 pp','题目95%CI','seed/题目95%CI'],contrasts)+'\n')
parts.append('baseline全文相同率、按原对错分组的救回/丢失率及逐文件哈希见[完整迁移诊断](results/math_validation_transfer.json)。文本相同率只能检查输出是否被原样复用，不能证明两个运行的内部执行等价。\n')

if (S/'audits/validation_case_mix_plan.json').exists():
    case_plan=source('audits/validation_case_mix_plan.json')
    contract=source('audits/validation_case_mix_contract.json')
    assert contract['passed'] and contract['script_sha256']==sha(S/'scripts/validation_case_mix.py')
    rows=[]
    for mode in ['raw','strict']:
        case_result=source(f'results/validation_case_mix_{mode}.json')
        assert case_result['primary_plan_sha256']==sha(S/'results/confirmation_plan.json')
        assert case_result['plan_sha256']==sha(S/'audits/validation_case_mix_plan.json')
        assert case_result['script_sha256']==contract['script_sha256']
        for grouping,v in case_result['results'].items():
            if v['identifiable']:
                rows.append([mode,grouping,f"{v['unadjusted_gap_pp']:+.2f}",f"{v['standardized_gap_pp']:+.2f}",ci(v['question_ci95_pp']),ci(v['seed_question_ci95_pp'])])
            else:
                rows.append([mode,grouping,f"{v['unadjusted_gap_pp']:+.2f}",'有分层缺少共同支持','—','—'])
    parts.append('### 相同题型、难度构成下的1500与3500差距\n\n这项诊断在主候选五seed完整MATH已完成后追加并留有[声明](audits/validation_case_mix_plan.json)。对同一主候选，在每个分层分别计算1500及3500上的adapter减baseline效应，再以完整5000题的分层占比作为共同权重。表中差为选模1500的效应减其余3500；正数表示前者收益较大。题目在各分层及两部分内独立重采样，保留每题五个模型的对应关系；第二种区间再共享抽取训练seed。观察到的分层占比固定，区间没有校正适应性选模或新增诊断的多重比较。\n\n'+table(['评分','共同构成','未调整差 pp','调整后差 pp','题目95%CI','seed/题目95%CI'],rows)+'\n\n这只能检查已标注题型和难度的粗粒度构成差异；调整后差距若保留，也可能涉及同一分层内的内容、难度与抽样差异，不能单独识别选模偏差。若任何分层在某一部分没有题目，该项不估计，也不通过删层获得结果。\n')

r=source('results/final_math_strata.json');assert r['partition_conservation_passed']
parts.append('### 全部题型与难度分层\n\n分层方案在首个SFT完整5000结果前登记；每种划分覆盖全部5000题，按题数加权必须精确还原总体效应。下表展示主候选的全部七类题型及五级难度；其他家族、训练题数/目标token分布与原始计数在[JSON](results/final_math_strata.json)及[CSV](results/final_math_strata.csv)。这些未校正的描述性区间不用于另选主模型，也不单独识别原因。\n')
for mode in ['raw','strict']:
    rows=[]
    for x in r['rows']:
        if x['scoring']!=mode or x['family']!=primary: continue
        rows.append([x['field'],x['category'],x['n'],x['base_correct'],seq(x['per_seed_correct']),
                     f"{x['delta_pp']:+.2f}",ci(x['seed_question_ci95'])])
    parts.append(mode+'评分。\n\n'+table(['划分','类别','题数','Baseline','五seed正确','均值Δ pp','seed/题目95%CI'],rows)+'\n')

if (S/'audits/minerva_pendulum_reference_sensitivity_plan.json').exists():
    pendulum=source('results/minerva_pendulum_reference_sensitivity.json')
    reference=source('audits/minerva_main_warning_preview_review.json')
    assert pendulum['script_sha256']==sha(S/'scripts/analyze_minerva_pendulum_sensitivity.py')
    for p,digest in pendulum['source_sha256'].items(): assert sha(S/p)==digest,p
    rows=[]
    for mode,v in pendulum['results'].items():
        for label,key in [('原272题','original_ood_macro'),('移除摆题132','ood_macro_excluding132')]:
            e=v[key];rows.append([mode,label,f"{e['delta_pp']:+.3f}",ci(e['question_ci95']),ci(e['seed_question_ci95']),seq([round(x,3) for x in e['per_seed_delta_pp']])])
    parts.append('### Minerva摆题参考方程的独立疑点\n\n完整Minerva主模型比较warning中，六条涉及index132。它的参考方程重力项为负；匹配的[MIT原题Q2b及图，PDF第2页](https://ocw.mit.edu/courses/2-04a-systems-and-controls-spring-2013/58bc43322a2253e2d825793381e0688a_MIT2_04AS13_ProblemSet1.pdf)显示下悬摆、从向下竖直线计角、水平外力。图已实际查看。在角度及力均取向左为正时，切向投影给出mlθ̈+mg sinθ=f cosθ，重力应为恢复项。原题匹配基于文字与图的推断，本地数据未附显式原题标识；因此保留参考疑点，不擅自改成另一gold。六个被warning标记的实际输出仍有独立于该符号的明确错误，例如水平力投影缺失、质量或长度缺失，或最终方程完全不含输入；逐条反例已记录。\n\n在查看这些warning后声明移除该题的事后敏感性，保留原272题结果，并以271题重算Minerva准确率和三个任务的等权宏平均。它不能改变完整MATH未通过的主标准，也不证明其他参考无误。\n\n'+table(['评分','Minerva范围','OOD宏均值Δ pp','题目95%CI','seed/题目95%CI','五seedΔ pp'],rows)+'\n\n[声明](audits/minerva_pendulum_reference_sensitivity_plan.json)、[逐条warning及参考复核](audits/minerva_main_warning_preview_review.json)、[独立敏感性](results/minerva_pendulum_reference_sensitivity.json)。\n')

for review_path,title in [
    ('audits/minerva_explicit_unit_review.json','Minerva实际单位表示边界'),
    ('audits/math2854_all_final_review.json','极小阶乘参考的实际输出'),
    ('audits/final_transition_review_decisions.json','固定抽样的成功与失败推导')]:
    r=source(review_path);assert r['all_reviewed']
    assert isinstance(r['report_markdown'],str) and r['report_markdown'].strip()
    parts.append('### '+title+'\n\n'+r['report_markdown']+'\n\n[完整逐例复核]('+review_path+')。\n')

report+='\n'+'\n'.join(parts);path.write_text(report)
sp=S/'audits/final_report_sources.json';record=get('audits/final_report_sources.json')
record.update(time=time.time(),before_diagnostic_append_report_sha256=before,report_sha256=sha(path),
              diagnostic_append_script_sha256=sha(__file__),diagnostic_source_sha256={p:sha(S/p) for p in sources})
write(sp,record)
print(json.dumps(dict(report=str(path),sources=len(sources),report_sha256=sha(path))))
