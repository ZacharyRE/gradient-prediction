# 文献与官方实现来源

访问日期：2026-09-08。以下均为原始论文、作者研究文章或官方文档；外部结果不是当前仓库的实验结果。

1. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**. 2021, ICLR 2022. https://arxiv.org/abs/2106.09685 。原始 LoRA 在所研究的模型和任务上能够接近或超过 full fine-tuning；这反驳“低秩适配从原理上不能用于监督微调”，但不能保证本仓库数学生成的结果。
2. Dettmers et al. **QLoRA: Efficient Finetuning of Quantized LLMs**. 2023, NeurIPS 2023. https://arxiv.org/abs/2305.14314 。使用冻结量化基座加可训练低秩适配完成 instruction tuning，研究覆盖多种数据及模型。模块覆盖是关键变量；不能从量化指令模型的对话评测直接推出本任务答案准确率。
3. Biderman et al. **LoRA Learns Less and Forgets Less**. 2024, TMLR. https://arxiv.org/html/2405.09673v2 。以 Llama-2-7B 比较 code/math CPT 与 IFT，并分别调 LR。Math IFT 中 r64 约62.4%、r256约63.4%，full最高约64.2% GSM8K；CPT 的容量差距更持久。起点、训练量、任务均与本仓库不同。说明低 rank 可能有限制，也说明“LoRA 不适合 SFT”过度概括。
4. Schulman and Thinking Machines Lab. **LoRA Without Regret**. 2025-09-29. https://thinkingmachines.ai/blog/lora/ 。涵盖 Llama3/Qwen3、rank与LR、batch及模块实验，建议关注MLP覆盖与独立LR。尤其注意其主要SFT比较指标是log loss，部分数学结果是RL reward；不能把CE接近解读成所有SFT生成能力相等。它与前项研究在任务、训练量、rank、评价指标不同，不应只选支持某一立场的结论。
5. Hugging Face. **Fixing Gradient Accumulation**. 2024-10-16. https://huggingface.co/blog/gradient_accumulation 。明确变长token任务应对整个有效batch的非padding监督token归一化；微批均值的平均通常不等价。本文仓库缺陷的直接依据仍是本地梯度数值复验。
6. vLLM. **Reproducibility**. stable documentation, accessed 2026-09-08. https://docs.vllm.ai/en/stable/usage/reproducibility/ 。默认不保证复现；batch invariance用于减小调度/批次影响。文档的硬件与版本边界意味着本研究仍须实际重复生成检查。
7. vLLM. **Batch Invariance**. stable documentation, accessed 2026-09-08. https://docs.vllm.ai/en/stable/features/batch_invariance/ 。该能力仍标注beta，记录配置本身不能替代实测。
8. Hugging Face PEFT. **LoRA**. accessed 2026-09-08. https://huggingface.co/docs/peft/en/package_reference/lora 。解释target_modules、all-linear范围及rank/alpha设置。本机实验显式列举七类transformer线性矩阵；不包含输出头、embedding、norm和bias，因此另设dense与full分支。
9. Qwen Team. **Qwen2.5-1.5B-Instruct model card**. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct 。当前基座已经instruction tuned，需区分“从base训练指令能力”与“继续改变一个已有能力的Instruct模型”。本机固定snapshot为989aa7980e4cf806f80c7fef2b1adb7bc71aa306。
10. Wu et al. **On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification**. arXiv v3, 2026-02-27; ICLR 2026. https://arxiv.org/html/2508.05629v3 。提出DFT，用停止梯度的token概率给CE加权。数学任务存在正向结果，也讨论事实知识任务限制。它支持把训练目标视为独立实验因素，但没有证明当前模型必须换目标；本研究不得把未执行的DFT宣称为修复。
11. **On the Role of Reasoning Patterns in the Generalization Discrepancy of Long Chain-of-Thought Supervised Fine-Tuning**. 2026, preprint. https://arxiv.org/html/2604.01702v1 。研究不同长CoT数据的loss与泛化差异。与本研究的短MATH参考分布不同，只作“参考似然不能自动替代生成评测”的旁证，不据此确认具体风格机制。

文献适用原则：明确区分CPT/SFT/RL、base/Instruct、loss/生成accuracy、单任务/OOD；不存在只凭LoRA名称就能迁移到所有设置的保证。上述来源在报告中采用编号引用；源网页的可变版本不替代本地依赖快照。
