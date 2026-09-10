# SFT 训练诊断研究

最终结果以[研究报告PDF](REPORT.pdf)、[网页报告](REPORT.html)或[Markdown报告](REPORT.md)为准。详细实验定义见[METHODS.md](METHODS.md)，原始计划见[PLAN.md](PLAN.md)。报告生成后，所有表格与图都可追溯至逐题记录。

目录内容：

| 目录 | 内容 |
| --- | --- |
| `results/training/` | 每条轨迹的参数、训练历史、样本曝光及checkpoint |
| `results/merged/` | 临时FP32合并模型的来源、哈希及清理记录；完成评测后不保留完整权重副本 |
| `results/evaluation128/` | 主要统一协议下的逐题生成、摘要及fingerprint |
| `results/evaluation/` | 初始并发32控制与未采用的重复输出，不混入主分数 |
| `results/selection.json` | 在确认题评分前冻结的模型选择 |
| `results/metrics.csv` | 所有完成的主要评分和配对统计 |
| `audits/` | 数据隔离、数值检查、历史重评分、案例、代码版本及协议修订 |
| `data/` | 本轮固定训练、开发、确认和诊断题集 |
| `snapshots/` | 本轮开始时的原有代码与报告快照 |
| `scripts/` | 训练、评测、统计、复验和报告生成脚本 |
| `logs/` | 命令、运行日志与定期GPU可用性记录 |
| `literature/` | 原始文献PDF、引用说明与下载哈希 |
| `figures/` | 可独立导出的PNG/PDF图及其CSV数据 |
| `.renderenv/` | 独立的文档/绘图环境，不改动训练依赖 |

复验单条已经记录的轨迹时，使用新的输出目录；以下示例先检查指定GPU是否有足够空闲资源，再复制小型输入/代码文件，按原manifest重新训练和生成，不覆盖原结果：

```bash
.venv/bin/python research/sft_diagnosis_20260908/scripts/reproduce.py \
  --source-run lora_qkvo16_lr1e-5_s43 \
  --destination research/sft_reproduction_qkvo \
  --gpu 4
```

`--datasets dev math_confirmation gsm_diagnostic`可复验全部指定题集；这些题已经在本研究中使用，复验不构成新的独立测试。`--source-run`可从`results/training/`中任选完整轨迹。更换软件或硬件可能改变数值结果，先核对`audits/environment.txt`。

`evaluation_pool*.py`、`precision_controls.py`等保存了本次实际分阶段调度，其中部分交接记录包含本次进程号；不要把这些历史协调脚本直接当作跨机器的一键入口。可移植的单轨迹复验入口是`reproduce.py`，基础训练/评测入口是`train_control*.py`和`evaluate_control128.py`。所有检查均可通过对应脚本与审计记录追溯。

checkpoint主要保存模型/adapter，并不包含精确续训所需的全部optimizer/RNG状态；新的seed复验均从base重新开始。

LoRA长期保存`results/training/<run>/step<step>/adapter_model.safetensors`中的A、B及adapter配置，共用固定版本的基座。统一推理对照使用的完整合并权重是临时文件；`reproduce.py`每个checkpoint完成评测后自动清理，保留导出哈希。需要重新合并已有adapter时，运行`export_model.py <adapter目录> <新的导出目录>`；评测完成后可运行`cleanup_lora_exports.py <导出目录>`，导出目录必须位于该研究的`results/merged/`下。full/dense SFT对照的完整训练权重继续保留。
