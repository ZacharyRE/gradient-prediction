"""Append separately reviewed supplemental evidence; preserve original primary verdict."""
from common import *
def get(path):return json.loads((S/path).read_text())
def ci(v):return '['+', '.join(f'{x:+.2f}' for x in v)+']'
def yn(v):return '是' if v else '否'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                     *['| '+' | '.join(map(str,r))+' |' for r in rows]])
report=S/'REPORT.md';text=report.read_text();before=sha(report)
marker='## 补充：扩容与checkpoint平均的五seed对照';assert marker not in text
review=get('audits/expanded_supplement_final_review.json')
assert review['all_reviewed'] and review['gpu_work_terminal']
status=get('audits/expanded_unaveraged_supplement_status.json')
planpath=S/'audits/expanded_unaveraged_supplement_plan.json';plan=get('audits/expanded_unaveraged_supplement_plan.json')
assert review['plan_sha256']==status['plan_sha256']==sha(planpath)
parts=[marker+'''\n
原主方案的16个模型与成功标准保持不变。看到扩容seed43–46的开发结果、但在seed47结果与任何完整5000/OOD输出出现前，另登记五个扩容未平均epoch4模型。该补充没有新增训练；最初计划在原生复核完成后复用GPU1，原登记最晚09:00 UTC启动；实际原生rank64耗时高于早期rank16估计，在任何补充GPU输出前通过[独立时间修订](audits/expanded_supplement_execution_amendment.json)将启动上限延至10:30 UTC，14:00 UTC结束上限不变。随后用户明确将上限改为三张卡，实际依据[GPU2执行修订](audits/gpu2_additional_execution_amendment.json)，在GPU2先完成原生目标控制，再执行本组baseline/非零adapter复放及五个模型，原GPU1/GPU3主任务持续运行。所有实际跨卡复放必须逐项一致；冻结分析器沿用的gpu1文件名只是显式标记actual_gpu=2的兼容记录，实际证据在[GPU2复放](audits/expanded_supplement_gpu2_protocol_replay.json)。之后，在任何实际OOD输出前登记[共享队列修订](audits/balanced_ood_execution_runtime_amendment.json)：GPU2已开始的五模型MATH和GPU3的原16模型MATH均自然完成，再共同处理原16模型的OOD，最后处理补充OOD。新派发工作结束上限改为15:00 UTC，硬截止仍为15:16:40 UTC；这取代早期14:00 UTC的执行余量，未改变模型或统计标准。完整MATH/OOD评估的模型数因此最多为21，这一范围扩展单独留档，没有改写原16模型计划。它用于补全“旧/扩容数据 × 平均/未平均”的比较；区间为次要、未校正的探索结果，不替换原主检验。旧数据与扩容数据都固定四轮，因此扩容同时改变问题覆盖、总曝光和优化步数；这组五seed家族差异不是单独控制覆盖的因果效应。将新增问题与同槽数旧题重复进行匹配的控制只在seed43完成，具体证据见前文。

[补充前置计划](audits/expanded_unaveraged_supplement_plan.json)、[权重冻结](audits/expanded_unaveraged_supplement_frozen.json)、[终态与复核](audits/expanded_supplement_final_review.json)。
''']
sources=['audits/expanded_unaveraged_supplement_plan.json','audits/expanded_unaveraged_supplement_status.json','audits/expanded_supplement_final_review.json']
if not status['all_complete']:
    parts.append('补充任务没有完成全部五seed的全部评估，终态为`'+status['status']+'`。已有部分输出仅作运行记录，不据此作一般性提升判断；原主方案的完整结果与结论见前文。\n')
else:
    a=get('results/expanded_unaveraged_supplement_analysis.json');assert a['plan_sha256']==sha(planpath)
    warnings=get('audits/supplement_scoring_warning_review.json');assert warnings['all_reviewed'] and warnings['plan_sha256']==sha(planpath)
    assert get('audits/expanded_supplement_gpu1_protocol_replay.json')['passed']
    sensitivity=get('results/expanded_supplement_math_sensitivity.json')
    extra='teacher7_expanded_epoch4'
    pairlabels={
        'teacher7_expanded_epoch4_minus_teacher7_epoch4':'扩容epoch4 减 旧数据epoch4',
        'teacher7_expanded_avg1234_minus_teacher7_expanded_epoch4':'扩容平均 减 扩容epoch4',
        'teacher7_avg1234_minus_teacher7_epoch4':'旧数据平均 减 旧数据epoch4',
        'teacher7_expanded_avg1234_minus_teacher7_avg1234':'扩容平均 减 旧数据平均'}
    parts.append('五个新增模型均完成完整MATH及四个OOD；实际GPU2重放baseline和固定非零adapter各500题，全文、评分、长度与停止原因均与GPU3记录一致。原16个模型的生成与评分没有重写。\n')
    for mode,result in a['results'].items():
        title='主评分，Minerva数值已修复' if mode.startswith('primary') else '严格评分，Minerva数值已修复'
        rows=[]
        for dataset,v in result['datasets'].items():
            f=v['families'][extra];x=f['vs_base'];rows.append([dataset,f"{v['base_correct']}/{v['n']}",'/'.join(map(str,f['per_seed_correct'])),f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95'])])
        parts.append('### '+title+'\n\n'+table(['集合','Baseline','五seed正确数','均值Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n')
        m=result['ood_macro'][extra]
        parts.append(f"数学OOD宏平均：五seed差为{'/'.join(f'{x:+.2f}' for x in m['per_seed_delta_pp'])} pp，均值{m['delta_pp']:+.2f} pp，题目区间{ci(m['question_ci95'])}，seed/题目区间{ci(m['seed_question_ci95'])}；每seed严格为正：{yn(m['every_seed_positive'])}。\n")
        rows=[]
        for pair,x in result['datasets']['math_reused5000']['contrasts'].items():
            rows.append(['MATH5000',pairlabels[pair],f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95'])])
        for pair,x in result['ood_macro_contrasts'].items():
            rows.append(['数学OOD宏平均',pairlabels[pair],f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95'])])
        parts.append(table(['集合','直接对照','均值Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n')
        parts.append('对应描述性条件：'+'；'.join(f'{k}：{yn(v)}' for k,v in result['descriptive_criteria'].items())+'。这些条件的列示不把补充检验升级为原主检验。\n')
        rows=[]
        for tolerance,v in result['supplemental_minerva_sensitivity'].items():
            m=v['ood_macro'];rows.append([tolerance,v['minerva_base_correct'],'/'.join(map(str,v['minerva_per_seed_correct'])),f"{m['delta_pp']:+.2f}",ci(m['question_ci95']),ci(m['seed_question_ci95'])])
        parts.append('Minerva评分敏感性。\n\n'+table(['评分','Minerva baseline','Minerva五seed','OOD宏均值Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n')
    parts.append('![扩容与平均的完整五seed比较](figures/expansion_factorial.png)\n')
    parts.append('### 相同MATH排除规则的补充敏感性\n\n沿用原主方案早已固定的3500/4997/4999/4996题索引，不根据补充输出选择排除项。\n')
    for mode,values in sensitivity['results'].items():
        rows=[]
        for label,v in values.items():
            x=v['vs_base'];rows.append([label,v['n'],v['base_correct'],'/'.join(map(str,v['per_seed_correct'])),f"{x['delta_pp']:+.2f}",ci(x['question_ci95']),ci(x['seed_question_ci95'])])
        parts.append(mode+'。\n\n'+table(['排除方案','题数','Baseline','五seed正确数','均值Δ pp','题目95%CI','seed/题目95%CI'],rows)+'\n')
    parts.append('[完整补充分析](results/expanded_unaveraged_supplement_analysis.json)、[MATH敏感性](results/expanded_supplement_math_sensitivity.json)、[评分warning复核](audits/supplement_scoring_warning_review.json)、[实际GPU2协议重放](audits/expanded_supplement_gpu2_protocol_replay.json)、[保留旧文件名的兼容记录](audits/expanded_supplement_gpu1_protocol_replay.json)。\n')
    sources+=['results/expanded_unaveraged_supplement_analysis.json','results/expanded_supplement_math_sensitivity.json','audits/supplement_scoring_warning_review.json']
report.write_text(text+'\n'+'\n'.join(parts))
sp=S/'audits/final_report_sources.json';record=get('audits/final_report_sources.json')
assert record['report_sha256']==before
record.update(before_supplement_report_sha256=before,report_sha256=sha(report),supplement_append_script_sha256=sha(__file__))
record['source_sha256'].update({p:sha(S/p) for p in sources});write(sp,record)
print('Appended reviewed supplement; original primary verdict preserved.')
