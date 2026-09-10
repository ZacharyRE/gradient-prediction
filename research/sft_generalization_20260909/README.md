# LoRA SFT 调研报告与证据附件（2026-09-09）

[阅读 Markdown 报告](REPORT.md) · [下载 30 页 PDF](REPORT.pdf) · [独立 HTML](REPORT.html)

本轮研究发现：原始参考解答监督与当前训练协议的组合造成明显退化；替换监督目标可恢复大量准确率，但仍没有达到预先规定的多 seed、跨数学任务稳定提升标准。主方案五个 seed 的 MATH5000 平均增益为 **+0.804 个百分点**，95% seed/题目区间跨零；数学 OOD 主宏平均为 **−1.914 个百分点，五个 seed 全负**。这些结果不能证明 LoRA 普遍无法进行 SFT；修复后的残余瓶颈尚未唯一识别。

## 本目录包含什么

- **报告**：PDF、Markdown、HTML；最终 REPORT 三种格式与本地交付文件逐字节一致。
- **方法和复现说明**：[METHODS.md](METHODS.md)、[REPRODUCE.md](REPRODUCE.md)、[冻结确认方案](results/confirmation_plan.json)。
- **实验与证据**：[实验清单](results/experiment_inventory.json)、[主结果](results/confirmation_analysis.json)、[严格评分敏感性](results/confirmation_analysis_strict.json)、[OOD 逐条复核敏感性](results/ood_manual_scoring_sensitivity.json)，以及报告引用的统计和审计文件。
- **图表和源数据**：[figures](figures/) 中的图片、矢量版本及已有的 CSV/来源记录。
- **源码快照**：[scripts](scripts/) 保存本轮脚本，便于检查训练、评估和统计实现。
- **前轮证据**：报告引用的前轮研究文档和小型结果附件保留原相对目录；不是重新执行的实验。

## 发布范围与复现边界

这是报告与证据发布包，不是可直接从头运行的完整实验归档。模型权重、LoRA checkpoint、optimizer 状态、完整训练/评测数据、完整逐题预测、运行日志和下载的文献正文留在原工作区；仅包含报告明确引用的少量案例审查文本。引用到这些未发布材料的 Markdown 链接已标为“仅原工作区保留”。JSON 中的绝对路径、源码中的本机路径和历史截止保护按原始记录保留，不能作为下载地址或直接运行的承诺。

独立复算全部统计或重训需要另外准备原始资产、固定版本的依赖和新的运行目录，并按照 REPRODUCE.md 处理历史时限与路径。当前包支持阅读、逐 seed 统计检查、图表数值检查和方法审阅，不声称只靠本包即可完整复现。REPRODUCE.md 中的命令是历史研究说明，不要直接启动已经结束的队列。

[原研究目录说明](RESEARCH_README.md)和[最终交付检查](audits/final_delivery_check.json)描述的是完整本地归档。发布时只调整少数辅助文档中的本地资产链接，并新增本说明；没有修改报告结论、数值、冻结方案、统计结果或 PDF。逐文件原始/发布 SHA-256 与省略清单见 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。

下载仓库后，可在仓库根目录核验此发布包：

```bash
sha256sum -c research/sft_generalization_20260909/SHA256SUMS
```
