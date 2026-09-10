# 09:33 UTC外部正结果补查

以下只记录新检索材料与适用范围，不改变已冻结实验，也不把未经本地复现的公开分数计为本轮证据。原页面与代码文本哈希见[sources](sources.json)。

- [Critical Token Fine-Tuning v1](https://arxiv.org/html/2510.10974v1) Table6：Qwen2.5-3B的MATH分数为未微调34.6、LoRA SFT41.9、LoRA CFT49.0。它用GSM8K自产正确解答，对每个token用第二/第三候选替换并greedy续写，仅监督两条续写都错误的位置。论文采用多学习率/批大小搜索，并报告各方法较好配置；没有本地1.5B-Instruct五seed证据。该方法提供了尚未测试的token信用分配假设，本轮未实现它；不能由论文声称“全token监督导致本地残余失败”。一个位置未直接计入loss也不保证该位置输出熵不变，因为参数由其他位置共享更新；有限两条续写失败也不是所有替代推导不可能的证明。
- [同1.5B-Instruct公开GSM8K仓库](https://github.com/Man-in-the-Mirror-05/SFT-vs-GRPO-Qwen)报告SFT从32%到40%，但实际只评100题、256新token、随机解码；已检查其eval代码，问题抽样seed固定，生成使用do_sample=True，没有报告多个训练seed及配对不确定性。它不能代替本地MATH+OOD确认，更不能把标题中的提升当成满足本任务的证据。只读代码，未执行。
- [DuoNeural模型卡](https://huggingface.co/DuoNeural/Qwen2.5-1.5B-SFT-ArchonLatentGeo)的同1.5B-Instruct小数据LoRA结果，GSM8K flexible从0.5148降至0.4162，strict从0.3169升至0.4693。模型卡把此归为格式变化，但没有在所读卡片提供逐题人工语义审计或多个训练seed；两个提取分数的方向不同本身不能证明数学能力提升，也不能证明下降完全由格式造成。本地已有严格评分、格式/截断上界与具体错误反例，不能用该卡片的解释替代这些证据。

原研究的full/dense对照已存在于[前阶段报告](../../../sft_diagnosis_20260908/REPORT.md)，未重复训练。它们主要使用raw目标与较早算术协议；本轮生成目标、SDPA条件下没有完整full-vs-LoRA五seed矩阵。旧full失败只能约束旧设置，不能证明当前LoRA已达到full的最佳可达上限。
