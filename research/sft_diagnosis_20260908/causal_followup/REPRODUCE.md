# 冻结方案与复现入口

这里复现实际做过的目标文本干预。能力结论和独立确认结果以 [REPORT.md](REPORT.md) 为准；配置选择记录为 [confirmation_frozen.json](results/confirmation_frozen.json)。

## 具体修改

主要修改是训练 `solution` 字段的来源：使用 self_final.jsonl（仅原工作区保留：`data/self_targets/self_final.jsonl`） 中通过答案验证的基座自产解答，替代 raw_final.jsonl（仅原工作区保留：`data/self_targets/raw_final.jsonl`） 的原始参考解答。两份文件的 1175 道问题及其顺序完全相同。没有用 dev 或确认集的答案生成训练目标。

主方案保留原始 eager attention。QKVO LoRA r16/alpha16、dropout0、有效 batch16（microbatch4 × accumulation4）、LR1e-5、AdamW FP32 状态、weight decay0.01、clip1。先在 FP32 基座上初始化 A/B，再仅将冻结基座转换为 BF16；训练使用 BF16 autocast。这样同一 seed 的四格对照起点逐张量一致。

实际训练一轮 1175 条、74 个 optimizer steps，最后一个有效 batch 为 7 条。LR warmup 按 64 条曝光，cosine 的计划 horizon 是 8 轮；第一步 LR0，此后正常更新。一轮主动停止，不把日程改成一轮 cosine。种子固定 43、44、45，主交付 adapter 使用最先预设的 seed43。

## 从头训练

在仓库根目录执行，`--name` 必须是尚不存在的新目录名。入口验证冻结的脚本和数据哈希，输出只含 LoRA adapter 的 checkpoint；GPU 参数当前只接受用户允许的 1 或 3。

```bash
cd /mnt/micron/yixiaore/adapter
.venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/run_target_recipe.py --name reproduced_self_eager_s43 --gpu 1 --seed 43
```

只打印核验后的命令而不占 GPU：

```bash
.venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/run_target_recipe.py --name preview_self_eager_s43 --gpu 1 --dry-run
```

匹配的原始目标对照，仅多加 `--target raw`。`--attention sdpa` 会启用明确的 SDPA math 后端，用于四格消融；它不是主修复所必需，也不能单独保证准确率改善。

```bash
.venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/run_target_recipe.py --name reproduced_raw_eager_s43 --gpu 3 --seed 43 --target raw
```

固定基座是本机 snapshot：

```text
/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

当前研究主 adapter 在 factor_self_qkvo_b16_lr1e-05_eager_s43/exposure1175（仅原工作区保留：`results/dtype_training/factor_self_qkvo_b16_lr1e-05_eager_s43/exposure1175`）。训练入口独立复跑得到的 224 张 A/B 与它完全相同，最大误差 0：[eager 复现审计](audits/recipe_eager_training_reproduction.json)。SDPA 支路也做过同样的精确复现：[SDPA 复现审计](audits/recipe_training_reproduction.json)。

## 生成评测

下面使用固定 BF16 vLLM、未合并 adapter、greedy、2048 新 token 上限、TRITON、batch invariant。把虚拟环境的 `bin` 加入 PATH，使本机 vLLM 的编译辅助程序 `ninja` 可见。

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/evaluate_bf16_512.py --name reproduced_self_eager_s43 --dataset dev --adapter research/sft_diagnosis_20260908/causal_followup/results/dtype_training/reproduced_self_eager_s43/exposure1175
```

新的复现使用新名字，保留原有逐题输出和 protocol fingerprint。`dev` 是已复用的开发集；`math_confirmation_new` 是本次冻结后的确认集，研究完成后再次运行它属于复核，不能再称为新 holdout。

不要把 HF BF16 eager、HF SDPA、FP32 vLLM 的分数直接混比。也不要因同为 BF16 就假定算术相同。当前 128 并发与 512 并发的 base/非零 LoRA 共 1000 条完整输出已逐字核对一致。[批大小核查](audits/bf16_batch512_equivalence.json)

## 重构训练数据和统计

从保存的生成候选与人工排除清单重新构造两份训练文件，并验证精确哈希：

```bash
.venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/rebuild_final_targets.py
```

加 `--output 新目录` 可另存重构文件；不会覆盖原始数据。生成、自动筛选、人工检查的原始证据在 self_targets（仅原工作区保留：`data/self_targets`）；只有 12 条推理经过人工完整审查，不能将全部自产推理称为过程认证的数据。

```bash
.venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/analyze_target_replication.py --dataset math_confirmation_new
.venv/bin/python research/sft_diagnosis_20260908/causal_followup/scripts/confirmation_sensitivity.py
```

统计入口检查确认集、基座、adapter 和生成配置与冻结清单一致。逐题重采样保留三个 seed 之间的配对，另报告跨 seed 的敏感性区间。

## 文件与环境

LoRA checkpoint 保存 A/B、adapter 配置及 tokenizer；推理仍需固定基座。`resume_latest.pt` 另存优化器/RNG/曝光位置，供研究续训恢复，不是完整基座。这里未提供一键 resume CLI，不把保存了状态说成已验证任意中断位置续训。

早期 FP32 统一推理使用的合并模型仅为临时导出，评测后清理。full-SFT 对照更新了完整权重，其完整模型是独立对照证据，不能只用 LoRA A/B 表示。

本机验证环境：Torch2.13.0+cu130、Transformers5.15.1、PEFT0.20.0、vLLM0.27.1、NVIDIA H200。复现脚本依赖研究目录内保存的 trainer/scorer 源码快照和固定数据。`frozen_sources` 保存确认前的入口源码；各训练目录保留对应版本、manifest、历史与数据顺序。

旧的多 GPU 调度脚本仅记录早期执行方法；本次资源限制后的入口是 `run_two_gpu_queue.py` 和上述只允许 GPU1/3 的复现命令。GPU7 已排除。
