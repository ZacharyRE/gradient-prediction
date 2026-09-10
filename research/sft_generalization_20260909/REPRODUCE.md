# 复现入口（本轮评估、统计与复核已完成）

在仓库根目录运行。使用现有 `.venv`；绘图和报告渲染复用前次研究的 `.renderenv`。本研究初期使用GPU1/3，用户授权三张卡后增加GPU2，每张卡同时一个GPU worker。实际共享OOD调度见文末执行修订；下列GPU1/3命令保留为独立复现示例。先确认本轮调度已经结束且所用卡空闲，再执行下面的手工 GPU 命令。

## 查看已有结果，不重复训练

```bash
.venv/bin/python research/sft_generalization_20260909/scripts/status.py
.venv/bin/python research/sft_generalization_20260909/scripts/analyze_dev.py
research/sft_diagnosis_20260908/.renderenv/bin/python research/sft_generalization_20260909/scripts/plot_development.py
```

`results/train_queue.json` 保存开发实验列表；`logs/commands.jsonl` 保存实际启动命令。每个训练目录的 `manifest.json`、`train_source.py`、`common_source.py` 是该次实际使用版本。研究期间增加的诊断选项会形成新的源码版本，不能只用当前脚本哈希描述全部历史运行。

## 从固定基座重训一个开发配置

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python research/sft_generalization_20260909/scripts/train.py \
  --name reproduced_sample_all_lr1e5_s43 \
  --train-file research/sft_generalization_20260909/data/sample_all.jsonl \
  --seed 43 --lr 1e-5 --batch 16 --rank 16 --alpha 16 \
  --attention sdpa --modules qkvo --objective token_mean --epochs 8 --stop 4
```

新名字必须未使用；入口拒绝覆盖已有目录。`--epochs 8 --stop 4` 代表八轮 horizon、运行四轮，不等于四轮 cosine。多解答/重复解答配对另用 `--epochs 2 --stop 1`，详见其预记录计划。当前保存最新 optimizer/RNG，已另行完成从原teacher seed43 epoch4继续至epoch8的分支；续训不是独立seed。

脚本包含本轮24小时截止保护。研究截止后的独立复现，应在复制出的研究目录中明确登记新的时限与设备，而不是静默改写本轮原始记录。

## 生成评测

`scripts/evaluate.py --dataset <name> --manifest <json>` 接受 JSON 对象，将唯一模型名字映射到 adapter 的绝对路径；`null` 表示固定基座。默认严格复用当前 BF16、未合并 LoRA、TRITON、greedy2048 协议。`--base` 同时评测基座，`--queue` 仅供本研究开发调度使用。

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python research/sft_generalization_20260909/scripts/evaluate.py \
  --dataset dev --base --manifest research/sft_generalization_20260909/results/protocol_controls.json

.venv/bin/python research/sft_generalization_20260909/scripts/score_sensitivity.py --dataset dev
```

已完成的同名评测会跳过。若要独立重复生成，使用新的模型名字，保留原始输出。当前 `protocol_controls.json` 是协议复核清单，不是最终候选方案。

## 数据与派生模型

所有学生采样和教师生成原文保留在 `results/generation`。`build_targets.py` 使用固定 RNG、完整停止/长度/最后 boxed 验证、已有逐条排除清单。`audits/target_build.json` 和每个训练 manifest 保存实际数据哈希。保留生成输出再重建数据，无需再次占用 GPU 生成相同目标。

`interpolate_adapter.py` 仅改变普通 LoRA 的 alpha，实现固定基座与 SFT 权重插值；A/B 不变。它不属于新的训练运行，每个派生 checkpoint 的 `derivation.json` 记录源模型、源配置与缩放比例。alpha=0 的完整500题文本与 base 已逐条核对一致。最终冻结必须同时包含 adapter 权重和配置哈希。

## 统计与最终报告

开发统计包括全部候选的配对差异、bootstrap 和 Holm 敏感性；不能把被大量选择后的最好开发点当作独立显著提升。`analyze_confirmation.py` 需要先存在正式的 `confirmation_plan.json`，包含数据、完整 adapter 文件哈希及评测协议，随后才可做冻结确认分析。本轮已在2026-09-10 04:02:51 UTC创建并冻结该清单，主候选为teacher7_avg1234；原文件不可覆盖。

```bash
research/sft_diagnosis_20260908/.renderenv/bin/python research/sft_generalization_20260909/scripts/render_report.py
```

渲染命令须在最终 `REPORT.md` 写好后执行，生成独立 HTML 和 PDF。原始统计、图表 CSV、训练 checkpoint 与逐题输出仍是数值核查依据。

## 主方案冻结与严格敏感性入口

`freeze_confirmation.py --selection <明确选择的JSON>` 会在任何最终5000/OOD输出之前核对五seed配方、全部权重/数据哈希、baseline与非零adapter复放和输入协议，然后创建不可覆盖的最终计划。它不自动选择模型，也不启动GPU。全部最终模型使用同一既有 `evaluate.py`，按计划中的五个dataset逐一运行。

对每个最终数据集执行 `score_sensitivity.py --dataset <dataset> --tag confirmation_<dataset>`，保存所有最终模型的严格最后boxed向量。随后 `analyze_confirmation.py` 生成主指标（Minerva数值使用前置修复），`analyze_confirmation.py --strict` 生成独立敏感性结果，保留原评分。只有在对应输出完整且哈希与冻结计划一致时才能分析。主方案与对照之间的MATH和OOD宏平均差异都直接配对计算。


## 冻结后的最终任务交接与收集

以下入口必须先有不可覆盖的 `results/confirmation_plan.json`。GPU入口先设置对应停止哨兵，让原队列完成当前模型后自然退出，再检查本研究进程及可用显存；不会停止其他用户进程。分别在独立终端执行，不能同时运行新的探索GPU任务。

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python research/sft_generalization_20260909/scripts/final_evaluation_pipeline.py --mode benchmarks

PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python research/sft_generalization_20260909/scripts/final_evaluation_pipeline.py --mode native

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  .venv/bin/python research/sft_generalization_20260909/scripts/collect_final_evidence.py
```

benchmark入口先补齐冻结模型缺少的开发输出，再评测MATH5000及四个OOD；不得依开发补充分数改变冻结选择。native入口用HF/PEFT的SDPA math、BF16基座/计算与FP32 adapter，对全部五个主模型及base运行相同500题。其前128题base要逐文本复放先前原生结果，且严格检查adapter加载张量。

CPU收集入口等待完整输出，执行严格评分、原始和严格确认统计、完整5000与旧1500重叠文本复放，以及原生/vLLM各自内部相对baseline的五seed比较。`audits/final_collection_status.json`的完成只表示产物收齐，仍须人工检查评分warning、判定标准及最终报告。严格原生评分使用只读symlink视图，不改写原始预测。

另外预登记的3500题敏感性严格取本轮开发验证1500题在完整5000中的补集；仍是历史复用材料，不是独立holdout。最终冻结脚本会检查实际索引、题目/答案对应关系以及所有相关文件哈希。

### 最后一次受限重复与自动冻结

扩容7B的四个额外seed由`audits/teacher7_expansion_seed_replication_plan.json`在训练前固定。`scripts/expanded_seed_replication_pipeline.py`等待旧队列自然退出后，仅执行这四项训练，保留`STOP_TRAINER`，每项结束后用原有有效更新平均脚本派生epoch1..4平均。

`audits/final_selection_rule.json`在这些seed训练前固定两候选的开发选择规则。`scripts/final_campaign_pipeline.py`等所有固定开发证据完整后，核对协议与严格评分，生成`audits/final_selection_decision.json`并调用原有冻结脚本；随后分别交接GPU3完整benchmark和GPU1原生后端，CPU只收集统计。它不会根据最终MATH/OOD成绩改配方。已运行的campaign不能重复启动；实际冻结、子进程PID、退出码由同名前缀的audit记录。

CPU补充分析可独立复算：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/sft_generalization_20260909/scripts/audit_loss_accuracy_alignment.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/sft_generalization_20260909/scripts/audit_final_statistics_contract.py
```

前者对齐64题reference CE与相同问题的自由生成；后者仅检查统计函数的数学不变量，不读取最终成绩。不要在最终冻结后覆盖被方案锁定哈希的审计文件；复算时使用另一个副本目录保留原始证据。

### Minerva数值向量与最终分析

最终分析不能只读取Minerva预测文件里的旧legacy correct字段。收集器在普通严格评分后运行CPU入口 scripts/score_minerva_numeric.py。

它要求真实冻结方案及其所有模型，创建不可覆盖的 audits/scoring_minerva_numeric_corrected.json。数值题用预先修复的科学计数法和相对比较，符号题保留原始/严格两种向量；后续 analyze_confirmation.py 自动使用该审计并检查哈希，同时报告旧评分和1%/5%容差敏感性。原始数据、生成文本、legacy评分仍完整保留。

audits/minerva_synthetic_fixture 系列及 audits/final_analysis_synthetic_fixture 是明确标注的CPU合成测试产物，里面的“预测”和“方案”不是模型实验或真正的最终选择。实际最终产物始终在本研究根的 results/confirmation_plan.json 与 results/evaluation/，不能把测试fixture计入模型成绩或独立seed。


### 单独的扩容未平均补充与原生目标对照（最初安排，后被文末修订取代）

主benchmark清单仍为16个模型。单独的 `audits/expanded_unaveraged_supplement_plan.json` 在任何最终5000/OOD输出前登记五个扩容未平均epoch4模型；实际权重及同配方核验在 `expanded_unaveraged_supplement_frozen.json`。GPU1先完成原主候选的原生500题，补充pipeline再完成自身baseline/非零adapter各500题重放及五个新模型的benchmark；原登记启动上限09:00 UTC；原生rank64实际耗时较早期rank16估计更长，因此在任何补充GPU输出前单独记录[执行时间修订](audits/expanded_supplement_execution_amendment.json)，将启动上限延至10:30 UTC，GPU结束上限仍为14:00 UTC，原主标准不变。当时的接续脚本为 `scripts/expanded_unaveraged_supplement_execution.py`，已在尚未启动GPU任务时退役；实际三卡及共享队列入口见后文。

`native_target_control_plan.json` 另规定在上述GPU1工作终结后，对两个matchedraw模型及原7B生成目标epoch4做HF/PEFT前128题，复用已有baseline和学生生成目标128题。原入口为 `native_target_control_pipeline.py`，原启动上限12:30 UTC、结束上限14:00 UTC；它后来在没有GPU输出时退役，实际对照已通过后文GPU2入口完成。它只检查单seed目标效应是否也存在于原生后端，不是新的五seed泛化检验。

所有GPU调度都有明确所有权交接和截止保护；当前研究中已启动的父pipeline不能重复执行。实际PID、进展、退出状态分别记录在同名前缀的audit与log中。可以只读查看：

```bash
.venv/bin/python research/sft_generalization_20260909/scripts/status_final.py
```

以下CPU分析需要各自对应的真实输出完整，并保留原主和补充的分工。最终文件已有哈希关联；独立复算应使用副本目录。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/analyze_expanded_supplement.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/analyze_supplement_math_sensitivity.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/prepare_supplement_warning_review.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/analyze_native_target_controls.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/analyze_final_math_strata.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/analyze_math_validation_transfer.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python research/sft_generalization_20260909/scripts/prepare_minerva_unit_review.py
```

原生目标对照的 `native_target128.jsonl` 与 `results/native_target_scoring_view/` 只是既有开发集前128题的评分视图；native输出用只读symlink，vLLM输出取完整500题文件的前128行。它们不是新增训练或OOD材料。`reference_ce_alignment_same_native64` 使用原先固定CE的同64题，补充HF自由生成计数。

`math_validation_transfer` 是看到前两个完整MATH主模型结果后登记的描述性诊断，使用早已固定的1500/3500分割。完整题型/难度分组也只作描述；它们不产生新的主成功标准。Minerva显式百分号/角度案例做题意与单位语境核查，不自动重写冻结分数。

合成测试还包括 `audits/expanded_supplement_synthetic_fixture`。其中21个模拟模型用于验证分析代码的分数传递、对照方向和零效应条件，不能算作真实模型、GPU重放或额外seed。

实际最终报告在原主结果、补充和原生目标对照全部达到终态并完成人工核验后，由 `build_final_report.py` 与 `append_supplement_report.py` 组装，再加入实际案例与分组解释。`plot_confirmation.py` 和 `plot_expansion_factorial.py` 只读取真实最终统计；绘图及HTML/PDF渲染使用上轮的renderenv。所有未执行或不完整的补充任务必须按实际终态报告，不得填造成功结果。

## 用户授权三卡后的实际执行

2026-09-10 06:27:57 UTC，根据用户明确的“可以3张卡”，增加此前空闲的GPU2。原GPU1主候选原生500题、GPU3主benchmark不被中断；仅退役尚未占用GPU的两个CPU等待器。GPU2顺序执行原生baseline/学生128题复放、三项固定原生目标控制、vLLM baseline/非零adapter各500题复放、五个冻结扩容未平均模型的完整benchmark。参见[资源与执行修订](audits/gpu2_additional_execution_amendment.json)。

该阶段调度器为 `scripts/gpu2_additional_execution.py`，通过 `scripts/gpu2_authorized_entry.py` 运行未改动的 `native_recheck.py` 和 `evaluate.py`。入口只在进程内替换资源守卫以允许GPU2，不修改原common文件、推理算术、模型、数据或统计标准。补充分析器的历史文件名 `expanded_supplement_gpu1_protocol_replay.json` 明确记录实际GPU2及实际审计路径/哈希；不代表执行过GPU1补充复放。

资源监控原记录保留在 `logs/resources.jsonl`；授权后由 `scripts/monitor_authorized_three.py` 写 `logs/resources_after_gpu2_authorization.jsonl`，增加本研究进程的实际卡号采样。最终资源审计按授权时间核对GPU2命令，并检查最多同时三张的采样记录。停止新监控使用 `STOP_MONITOR_AUTHORIZED3`，原 `STOP_MONITOR` 保持存在。这一阶段原定GPU2工作最晚14:00 UTC结束；后续共享队列修订将新派发工作结束上限改为15:00 UTC，研究硬截止仍为15:16:40 UTC。

### CPU证据收集的时间安排修订

为避免全部原生复核结束后才开始补充严格评分，07:40 UTC附近以[独立执行修订](audits/additional_collection_execution_amendment.json)替换尚未运行分析的CPU等待器。实际入口为`CUDA_VISIBLE_DEVICES='' .venv/bin/python research/sft_generalization_20260909/scripts/collect_additional_execution.py`。它在每个补充数据集五个模型的预测全部完成后提前运行原严格评分器；原补充分析器稍后验证并复用这些文件。固定案例抽样在主benchmark两种分析完成后准备，其他最终依赖与审阅要求不变。原九个诊断脚本、数值评分及统计程序的哈希均保持不变。该入口依赖本次单次调度状态；从头复现时应在独立输出目录执行相应底层步骤，不能覆盖本次审计。

## OOD共享队列的实际执行修订

09:15 UTC附近登记[运行时执行修订](audits/balanced_ood_execution_runtime_amendment.json)，以`scripts/balanced_ood_execution_v2.py`管理剩余GPU2/3评估。两卡原有MATH推理子进程持续运行；仅暂停CPU父调度器，等待子进程正常退出、全部汇总存在和两次GPU空闲检查后退役父调度器。GPU2既有1000条vLLM复放重新逐项核验并复用，五个已开始的补充MATH评估自然完成后标记复用。随后两卡共享原16模型的12组OOD任务，补充五模型的4组OOD任务排在后面；GPU1原生500题工作始终独立继续。

数值脚本、模型、数据、评分、统计与成功标准的冻结哈希不变。共享派发工作的结束上限为15:00 UTC，替代早期补充14:00 UTC余量，整体仍在15:16:40 UTC硬截止内。早期CPU协调器在没有派发任何GPU任务时退役，原状态与修订记录保留。

原主CPU调度器的benchmark子进程将因上述计划内退役记录-15，原主进程因其旧全零检查而退出非零；这些真实返回码保留。最终运行`scripts/reconcile_final_execution.py`，同时核对原MATH子进程退出0、实际替代OOD全部完成、原生和收集器退出0以及对应输出哈希。报告生成器要求这份独立核对通过。此脚本只生成新的核对记录，不改写原执行记录，也不代表统计有效性判断。

### 追加的粗粒度构成诊断

`validation_case_mix.py --scoring raw`及`--scoring strict`分别计算主候选在1500/3500上的共同题型、难度和联合构成调整。需要相应真实完整输出；严格模式还需要原主严格评分文件。该诊断在主五seed完整MATH已知后登记，计划和合成数学不变量检查分别在`audits/validation_case_mix_plan.json`、`audits/validation_case_mix_contract.json`。输出拒绝覆盖。它是固定观察权重下的描述性分析，不是新的主检验；不可因某种分组区间较好就挑选该分组宣称因果机制。原九个自动诊断脚本及主统计未修改。


### 现有梯度日志与提前阅读单位案例

`analyze_training_gradient_logs.py`只读已保存的13个固定训练运行，按`training_gradient_log_plan.json`核对全部文件哈希，统计截至指定端点的每一步裁剪前范数。它是结果后的描述性核查，不新增训练或更换模型；7920步没有触发clip1。结果与逐epoch值在`results/training_gradient_log_analysis.json`和CSV中，输出拒绝覆盖。

`preview_minerva_unit_cases.py`可在21个固定Minerva预测文件全部完成后提前提取原来预先声明的百分号/角度答案案例。它不需要等待原生500题结束，也不运行评分，写入独立的`minerva_explicit_unit_preview_*`文件。原`prepare_minerva_unit_review.py`仍由既定CPU收集器在其依赖完成后执行。若使用提前阅读结果，最终复核必须逐组核对问题、最后答案框、模型、全文和文件哈希与原最终案例清单完全一致，再读取最终数值判断；预览本身不表示审阅完成。


### 已退出子进程读取的实际恢复

11:07 UTC附近，v2在首个MATH子进程正常退出后读取其`/proc/environ`被拒绝，旧逻辑将其误报为丢失。此时没有派发任何OOD任务，原错误状态和源码保留。[实际进程契约](audits/balanced_ood_zombie_read_contract.json)确认UID、PID出生时间、父进程及退出码0均可核验，并验证错误出生时间会被拒绝。实际继续执行入口为`scripts/balanced_ood_execution_v3.py`，依据[独立修订](audits/balanced_ood_zombie_read_amendment.json)恢复同一待执行队列；退出进程无需读取已失效的环境，存活父进程的环境检查不变。GPU1/2当时的推理均继续，MATH不重跑。最终执行核对必须包含这次真实错误及恢复链。


事后完整评分分歧复核：`audits/math_scoring_disagreement_review_plan.json`先声明全部47条选择与歧义边界，再逐条读全文并保存`math_scoring_disagreement_review_decisions.json`。运行`.venv/bin/python research/sft_generalization_20260909/scripts/analyze_math_manual_disagreements.py`只生成独立敏感性，不改原评分。额外index4060全21模型复核按独立声明保存；`scripts/analyze_math4060_followup.py`用冻结compare函数从逐题向量重算区间。`results/math4060_followup_point_sensitivity.json`是先前仅含精确点估计的中间文件，区间未写入；正式包含区间的结果为`results/math4060_followup_sensitivity.json`。两者都不能替换主成功标准。

语义复核修订：`audits/math_operator_semantics_revision.json`保留最初将boxed12一概判错的过强判断和旧结果/源码。正式47条审查现在有8个歧义；追加脚本对所有模型统一比较答案标记与字面约数算子两种解释，`results/math4060_followup_sensitivity.json`字段为`results_by_interpretation`。两脚本已按修订实际重算bootstrap，不只修改文案或平移区间。


## 本轮最终产物复算

本次GPU与CPU流程均已结束。请在独立副本目录复算，保留已有冻结哈希和单次写入审计；不要重新启动本目录的历史队列。最终主、补充、原生、评分分歧及单位审查都已完成。`reconcile_final_execution.py`已核对原benchmark CPU −15、native0、collector0与替代OOD完整输出；原执行返回码保留。

报告顺序为`build_final_report.py`、`append_supplement_report.py`、`append_diagnostic_report.py`，随后使用renderenv的`render_report.py`。新追加的`analyze_ood_manual_sensitivity.py`只读取人工复核决定与原始分数向量，分别计算74条分歧修正，以及另加3个百分数正确答案后的四种家族、歧义上下界；原`compare()`/`macro_compare()`保持不变。`finalize_ood_manual_review.py`、`bind_minerva_unit_review.py`和`bind_completed_warning_reviews.py`仅核对已实际阅读的案例与最终文件相同，不替代人工审查。

完整MATH47条分歧与方框解释的旧审查修订见对应audit归档；所有事后分析均独立记录，没有修改主模型或成功标准。报告Markdown、HTML、PDF与图表来源的最终哈希和排版/链接检查见`audits/final_delivery_check.json`。
