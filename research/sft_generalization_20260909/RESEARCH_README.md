# LoRA SFT 泛化研究

最终报告：[Markdown](REPORT.md) · [HTML](REPORT.html) · [PDF](REPORT.pdf)。报告包含全部五 seed、受控对照、未成功尝试、评分复核及可复查的原始证据。

本轮证明了原始参考解答监督与当前训练协议组合造成自由生成退化；替换目标能大幅恢复，但最终仍未获得跨 seed 稳定的数学泛化提升。主方案 MATH5000 平均 +0.804 pp，seed/题目95%区间跨零；数学 OOD 主宏平均 −1.914 pp，五 seed 全负。严格评分的 OOD 正增益主要受 baseline 正确无框答案失分影响；人工复核敏感性也没有变成正向收益。这不是 LoRA 普遍不能 SFT 的证明，修复后的残余瓶颈仍未唯一识别。

研究从2026-09-09 15:16:40 UTC开始，在24小时内完成。71个训练运行、19个派生模型已登记；21个固定模型均完成5000题MATH及878题数学OOD，另有全部五seed的原生后端复核。23个未执行的训练候选保留为pending，不计作实验。392个评估汇总包括开发、复放与诊断，不代表392个独立配方。所有GPU任务、CPU收集器、资源监控和截止保护进程均已结束。

初期仅使用GPU1/3；用户明确允许三张卡后增加GPU2。最多同时三张的实际记录见[最终资源审计](audits/resource_scope_final.json)。[原执行返回码与替代执行核对](audits/final_campaign_execution_reconciliation.json)保留正常交接导致的原CPU返回−15以及真实恢复记录，不改写为原进程全成功。

- [方法与外推边界](METHODS.md)、[复现入口](REPRODUCE.md)、[初始计划](PLAN.md)、[冻结主方案](results/confirmation_plan.json)
- [训练与评估清单](results/experiment_inventory.json)、[因果证据](CAUSAL_FINDINGS.md)、[文献](literature/NOTES.md)
- [MATH评分独立审查](SCORING_AUDIT.md)、[全部74条OOD评分分歧](audits/ood_scoring_disagreement_review.json)、[OOD人工敏感性](results/ood_manual_scoring_sensitivity.json)
- [最终状态](STATUS.md)、[历史工作记录](WORK_STATE.md)、[历史调度](SCHEDULE.md)、[开发阶段台账](EVIDENCE_LEDGER.md)

所有逐题数学复核由本助手执行，不是独立人类盲评。争议、错误初判修订、未完成分支及参考答案疑点均保留；原始预测、冻结评分和主要成功标准没有因观察结果而修改。
