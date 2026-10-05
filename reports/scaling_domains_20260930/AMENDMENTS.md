# 实施记录与协议补充

## 首次最终测试前：效率和主比较细化

- 最终主比较固定为13个计划模型/任务条件，各比较开发集选中的predictor策略相对Base，以及相对开发集选中的真梯度策略，共26项；三种predictor策略和三种真梯度方法在同一LR网格选参。细则见configs/FINAL_ANALYSIS.md。
- Oracle/Mean路径不挂activation capture hook；在线校准采标签时关闭仅用于研究的解析重建核验。核验保留在离线cache采集，避免把纯诊断成本算成部署必需成本。所有方法相同精度和SDPA math后端。
- 固定额外敏感性：1.5B Countdown/GSM8K/AQuA，32步warmup LR 3e-6/5e-5/5e-4以及零步初始化，分别相同数据预算拟合predictor并测静态梯度。与主选中状态相同时直接引用主结果。该补充仅在留足最终评测时间时执行，未完成不填零。零B初始化的A梯度结构性为零，须单独解释。
- 新任务的科学/布尔目标只有答案，数学目标保留已有推导；不人为生成参考推理。所有划分38491条参考序列的评分自检通过；这不等于独立验证所有源数据答案事实正确。

当前没有运行最终测试，也没有依据最终结果调整候选、评分或报告口径。

## UTC 04:45：首次开发生成启动失败修复

第一轮45条1.5B warmup轨迹均完成。vLLM首次启动的FlashInfer JIT未在PATH找到已安装的ninja，尚无任何生成结果。将现有虚拟环境bin加入子进程PATH，原配置、checkpoint、评分和数据不变；保留原错误日志及三阶段退出记录。重新启动监督程序，已完成warmup按完成标记跳过。

## 最终测试前固定重复任务

预设1.5B Countdown/AQuA/ARC-Challenge为重复代表，predictor seeds124/125、update seeds102/103。相同warmup/数据；沿用主seed开发集选中的predictor策略与真梯度策略及各自LR。收集后如剩余至少12小时就补fit，更新前若剩余至少8小时就做重复轨迹；这是资源规则，不能根据最终结果挑seed或任务。若未满足预算则明示未完成。仅重启尚在等待phase1的phase2监督程序，GPU评测进程不受影响。

## 最终测试前：成本与尝试目录

新增完整尝试索引以及按实际32步轨迹计算的离线摊销分析。未来fit summary单独记录训练/开发选择时间与held-out审计时间；模型训练或选参规则未变。已等待的phase3监督程序重启以载入这些CPU分析步骤，GPU评测未中断。

## 最终测试前：实现预定质量/时间曲线

在step32的策略/LR选择和最终manifest冻结之后，对选中predictor与真梯度策略追加step8/16的完整开发集生成，结合既有step0/32开发结果及实际online_seconds。仅描述开发曲线，不用于重新选择终点或配置，不访问最终测试选参。读取/配对/不可变选择的CPU构造数据自检通过。final_evaluate.py会先运行checkpoint开发readout，再执行完整final评测；原final候选与统计检验不变。

## 最终测试前：修正融合predictor算子的FLOPs漏计

实测发现当前Torch默认FlopCounterMode未注册aten._transformer_encoder_layer_fwd：CPU两层encoder示例默认计数为0，而稠密矩阵/attention应为483840 FLOPs。新增flop_accounting.py，为实际执行的融合encoder/nativeMHA注册shape公式，保持训练/计时的kernel路径不变。CPU融合与非融合(math attention)均恢复483840，nativeMHA为127232；输出数值一致到1e-6。全GPU方法profiling将保存算子覆盖，并拒绝未计数的不透明attention。当前尚无efficiency结果，未发生已报硬件加速结论的撤回或结果重跑。精度说明明确predictor为FP32、骨干BF16 autocast。

## UTC 05:50:12：最终测试前修正选择题原选项文本解析

开发输出核查发现，旧主解析器将“A)原选项全文”判为错误，即使字母和正文都明确匹配题目选项。这导致1.5B ARC-Easy/Challenge Base低估7/3题，OpenBookQA低估1题；部分所谓warmup提升只是格式变化。新增v2只识别整段单一选项字母及其对应原选项文本，允许空白/大小写/单个句尾句点差异，不猜测自由推理中的字母，错配正文和多答案会拒绝。8个针对性核验通过。

这是对原“开发前冻结parser”约定的明确修订：首次predictor训练及最终测试均未开始。完整旧dev输出/评分/选定起点存于provenance/grading_v1；在原生成上重算新主评分，重新选起点，保留legacy_correct及strict boxed作为辅助结果。11968条完整问题/输出/token数逐条保持不变；50个正确性标志修正。1.5B新Base：ARC-Easy109/128、ARC-Challenge103/128、OpenBookQA95/128。ARC-Easy新起点1e-4×8=110/128，其余起点未改变。

7B开发worker启动时已载入v1，保持GPU生成不中断，select_warmup会在worker结束后自动迁移其评分再选择。所有后续update/final生成直接使用v2；最终manifest冻结grader revision/hash，最终分析同时保留旧受限格式与strict boxed比较。

## UTC 06:06:21：完成统一格式核查，固定v3评分

同一轮开发核查发现AQuA存在大量boxed数值而非字母输出：Base的22个数值答案精确匹配唯一正确选项，旧解析漏计。v3在v2基础上接受boxed内容与唯一原选项文本的字面匹配；支持成对美元定界符、完整LaTeX text/mathrm/textbf包装，以及BoolQ true/false→yes/no。仅规范化空白、大小写及一个句尾句点；不做数值近似/符号等价猜测，不从推理中搜索字母，重复选项匹配拒绝。16个核验通过，包括歧义/错配/自由文本拒绝。

首个predictor拟合和最终测试仍未开始。再次保留v2中间输出/选择，全部1.5B开发重算为v3；相对最初v1的11968条完整输入/输出/token记录完全一致，297个正确性标志修正，legacy_correct和strict_requested_format保持原值。AQuA新Base72/128，最优非零74/128，按CE破同分选1e-4×8；BoolQ新最优106/128，仍5e-4×8；其他起点同v2。7B正在运行的v1生成将在完成后直接迁移v3，后续方法从共同v3起点比较。最终主检验族仍26，最终manifest冻结v3及grader hash。

## UTC 06:30：首次Adaptive更新的混合特征合批修复

校准样本包含真实g，prefix-only样本不含g；首次Adaptive预测把两者合批时，pack根据首条样本分配g并在后续样本报KeyError。predict_pair现明确去掉推理不使用的g，仅传特征；GPU构造回归核验确认无标签、混合标签、所有标签三种输入的预测逐位相同，改变g数值也不影响结果。失败发生在首次LoRA optimizer更新之前；保留失败目录及完整日志。已完成Oracle/Oracle20/Mean轨迹保留；初始Frozen轨迹存档并用统一推理代码重跑。未改变数据、超参数、评分或选择规则。

## UTC 07:19：FLOPs计数改为原生dispatch，GPU回放核验通过

首次完整GPU测量发现默认FlopCounterMode运行中的Oracle/Oracle20/Mean参数更新与不带计数器的运行不同（最大坐标差约1.99e-5，计时运行的梯度norm稳定）。核查安装的PyTorch源码发现默认计数器会调用func.decompose，并启用模块/反向追踪hook。为保留实际执行路径，新增NativeFlopCounterMode：仅对实际原生dispatch应用Torch既有FLOP公式与融合encoder公式，不进行算子分解，也不加入模块梯度hook。CPU已知融合encoder=483840、冻结线性链前向及输入反向=229376 FLOPs，原生计数前后数值逐位一致。

等待正在运行的SVAMP更新完成后，仅短暂持住CPU队列，串行重测Countdown效率，再自动恢复队列。旧计时、FLOPs、覆盖和GPU审计完整存于provenance/profiling_v1/1p5b_countdown，同时保留当时安装的Torch计数器源码。新GPU测量中5种方法的普通重复与计数回放optimizer更新均逐位一致，FLOPs总数与旧计数相同；见results/native_profile_replay_audit.json。训练、开发选择、评分和数据未改动；后续效率测量使用新原生计数器。

## 最终测试前：补充冻结全权重训练均值对照

前两个开发pilot中，静态训练均值的batch梯度cosine与predictor相当或更高，因此补充一个闭环廉价控制：M=ΣGᵀX/Σ监督tokens在共同warmup固定，每步用当前A/B得到sBᵀM和sMAᵀ，再执行相同AdamW。它保留LoRA因子的变化，但不使用当前batch特征、在线真梯度或predictor拟合；不是每步重新计算全训练集梯度。未来static采集顺便保留M，已清理缓存的早期条件单独重建。

独立输出目录_mean_control，仍32步和原4档更新LR，仅按完整生成dev选参。主26项比较与原选择不变；补充13项predictor−冻结均值比较，同时给出合并39项的保守Holm校正，不把缺项当零分。全部在首次最终测试前声明。原warmup静态敏感性优先；该控制在阶段启动时须至少剩11小时，剩8小时不再开新条件，以保留完整最终评测时间。配置见configs/mean_matrix_plan.json。


## Pre-final warmup-to-downstream diagnostic

2026-10-01 09:02:20 UTC: Extend the already declared static warmup sensitivity on Countdown/GSM8K/AQuA with Frozen and Oracle32 updates at a common fixed update LR (the original main selected predictor LR for that task), followed by full generation-development evaluation. Warmup states remain zero and32steps at3e-6/5e-5/5e-4, plus the main selected state as a reused reference. This directly checks downstream sensitivity at fixed update settings; it is not an additional search for a final-test winner. It leaves primary choices and multiplicity families unchanged. Start only with10h remaining; stop new states with8h remaining, before the resource-gated mean-matrix control. All real final tests remain untouched. Configuration: configs/warmup_closed_loop_plan.json.

## Pre-final descriptive gradient-norm distribution

2026-10-01, after the ARC-Easy static audit: Add norm quantiles, top-ceil(10%*n) norm mass and squared-norm energy fractions, and norm-mass effective sample size to the existing held-out gradient geometry analysis. ARC-Easy's mean individual norm being much larger than its median motivated this post-hoc diagnostic. It reuses preserved CPU vectors, creates no new GPU trajectory, and does not change training, hyperparameter selection, grading, or hypothesis-test families. Squared-norm concentration is distinct from vector-sum contribution or directional cancellation; observed task differences do not establish a causal explanation for predictor error.

At the same time, make descriptive bootstrap indices and random batch partitions deterministic per condition, shared across its predictor seeds. Previously one global RNG stream meant adding a condition changed later conditions' resamples, and each predictor seed used different batch partitions. This only replaces descriptive resampling groups; actual per-example metrics, training runs, model selection and final inferential tests are unchanged. The seed and scope are recorded in each gradient-controls result.

After the ARC-Challenge static audit, also add a constant-zero-output residual reference and a descriptive across-example centered association. Subtracting the same training mean from a weak prediction and its target can itself create a positive residual cosine: ARC-Easy's zero-output reference is about0.659 versus predictor0.661; ARC-Challenge's is about0.526 versus predictor0.500. Centered association subtracts each held-out matrix's own mean only to describe covariation; it does not fit or select a predictor using held-out gradients. These checks limit interpretation of the existing residual cosine and leave all GPU experiments and primary tests unchanged.

## UTC 11:15: final-evaluation execution priority

Before any final selection manifest or final generation exists, change only the execution order: prepare and freeze the selected step8/16 checkpoint entries, run all planned full final-test versions for both model sizes, then evaluate the already fixed intermediate checkpoints on development data. The earlier checkpoint-first order is superseded. The updated development-based runtime forecast motivates prioritizing the core final readout under the authorized24h deadline. No versions, inputs, selection rules, checkpoint choices, hypothesis families or resource deadline are removed or changed; intermediate curves remain part of the planned work. No final result can influence the checkpoint entries because those are fixed before final generation starts.

## UTC 11:26: Countdown development output diagnostic

After the complete7B Countdown development grid, Frozen at update LR1e-4 has7/128 correct and mean output20.49 tokens. Audit all52 existing Countdown development versions across both model sizes (24 update endpoints plus Base/Warmup per model), retaining original correctness and parser fields. All inputs and question order match the full development split. The high-LR7B Frozen outputs all stop normally, all contain boxed expressions, and80/128 satisfy the number-inventory constraint; none hits the output-generation cap. This is a post-hoc output-behavior description, not an isolated causal test of reasoning length. No regrading, new GPU branch, selection change or additional primary hypothesis. Script and source hashes are retained with the diagnostic.

## UTC 11:33: enforce the frozen input manifest

Final-readout review found that the analyzer checked each output against the current test file and evaluation-protocol hash, but did not yet enforce the data-manifest hash already saved at selection freeze. Add that check, plus matching test-file hashes and question counts to the frozen manifest; the final evaluator also checks hashes before starting any generation. This closes a provenance-validation gap without changing any data, predictions, grading, selections or statistics. All9 actual final-input files still match their original manifest hashes. The isolated CPU freeze/analyze fixture passes and now rejects both a modified data manifest and a jointly modified test file/evaluation-protocol hash, alongside the existing source/adapter/truncation/missing-example checks. No real final generation has run.

## UTC 12:00: logged predictor-checkpoint criterion diagnostic

The completed7B AQuA fit selects epoch5 by the prespecified individual factor error, while its saved training history records a lower whole-development-batch relative error at epoch82. Add a CPU-only summary of both criteria across all completed predictor histories, preserving the actual selected checkpoints. The alternate epoch is only a logged point: no alternate predictor is trained, selected, or evaluated, and its held-out/downstream behavior is unknown. This motivates a future test of criterion alignment, not a claim of downstream improvement. Whole-dev aggregate error is distinct from random batch32 error. No primary candidates, hypotheses, grading, or final choices change.

## UTC 12:44: relative-error weighting diagnostic

After the completed7B ARC-Challenge predictor fit reported a very large individual relative error, add a CPU-only audit using already preserved held-out gradient vectors. Reproduce the original A/B-balanced relative squared error, inspect the actual raw-factor denominator floor1e-12, and contrast the smallest-gradient half of examples with its squared-norm energy and whole-matrix Frobenius error. Include1.5B ARC-Challenge and7B AQuA for context. This post-hoc diagnostic neither changes the objective/checkpoint nor trains an alternate predictor; no downstream improvement from a new loss is claimed. Exact values and source hashes are saved in results/relative_error_audit.json, with the definition in scripts/relative_error_audit.py. All primary selections and test families remain unchanged.

## Pre-final: separate main-grid and warmup-sensitivity figures

As the first static warmup supplement completed, add its descriptive gradient figure (individual, random-batch32, and across-example centered cosine) with the fixed training-mean control and explicit main-start markers. Restrict the existing main update-LR drift figure to exact primary condition names: supplementary Frozen trajectories use the same model/task and would otherwise be merged into those main curves. At this change no supplementary update trajectory has run, so no previously published main curve contained mixed starts. Saved data, training, selections and tests are unchanged; warmup downstream curves remain a separate figure.

## Pre-final presentation: distinguish activation and reconstructed LoRA metrics

After the zero-warmup AQuA fit, extend the gradient tables and warmup figure with the activation cosine already recorded by every original fit. GSM8K and AQuA show opposite directions of change between activation and reconstructed LoRA cosine when comparing zero warmup with the main selected start. The models, target gradients, and LoRA factors all differ across these states; the presentation therefore keeps both metrics and does not claim an isolated causal effect of factor reconstruction. No new metric definition, rerun, checkpoint choice, or inference family is introduced.

## UTC 13:43: final request queue with unchanged active capacity

Before final selection freeze and before any final generation, increase only the final evaluator request submission chunk from256 to1024. Keep max_num_seqs256, KV caches48/64GiB, full input prompts, max_new_tokens2048, greedy seed42, batch-invariant mode, adapters, scoring and all candidate families unchanged. The installed vLLM LLM.generate implementation accepts a pending request queue, runs the scheduler until requests finish, and returns outputs sorted by original request ID; the scheduler retains the active-sequence cap. This allows waiting requests to fill vacated slots instead of waiting for every request in a256-item submission to finish. Development/checkpoint evaluation remains unchanged. Larger output checkpoint intervals are explicitly recorded; no separate speedup factor or cross-queue bitwise identity is claimed. Installed source copies and hashes are preserved in provenance/final_evaluation_scheduler. The time forecast remains the unscaled development-rate extrapolation.

## UTC 13:50: matched descriptive resampling across warmup starts

After all12 static warmup states completed, extend the condition-stable diagnostic RNG grouping so that a task/model main state, its predictor seeds, and its supplementary warmup starts share bootstrap indices and random batch32 partitions. Previously each warmup suffix had a different deterministic seed. Require identical audit source hash, sample count, and ordered supervised-token counts; collection uses the same deterministic length-sorted source order. Preserve the previous script, results and PDF in provenance/warmup_resampling_v1. Actual comparison verifies all19 primary/repeat result rows retain their prior values (apart from clarified scope text); all11 supplemental states retain all unresampled geometry/per-example/residual/centered metrics. Only supplementary descriptive resampling changes. Training, held-out gradients, method selection and final hypothesis families are unchanged. Verification: results/warmup_shared_resampling_audit.json.

## 2026-10-01 14:52 UTC — external GPU0 occupancy and pre-final resource recovery

External processes572797/572798/572799 started14:43:26UTC and occupy about94GiB onGPU0. The7B Countdown MeanM updates completed while sharing the device; its following evaluation failed at initialization because free45.51GiB was below the104.85GiB startup request. No model was loaded or prediction generated by that failed evaluation. Its protocol/logs/phase3 error/source files and resource snapshot are archived under provenance/failures/external_gpu0_occupancy. No external process was stopped.

Resume only pending work using STUDY_EVAL_KV_GIB=16 and STUDY_EVAL_MEMORY_FRACTION=.25 for subsequent MeanM/final/checkpoint evaluation. Keep active cap256, full prompts, max_new_tokens2048, seed42, greedy/batch-invariant decoding, adapters, all data, selection rules and graders unchanged. Final request submission size1024 remains as previously declared; dev/checkpoints remain256. Actual per-stage protocols record resources. This supersedes the prior48/64GiB execution allocation, before any final freeze/test. The failed empty protocol is archived before recreating it; completed outputs are untouched.13 primary controlled timing/FLOPs cases and all primary update trajectories finished before the external processes started; later7B MeanM trajectory timings and evaluation walltime are shared-device observations, not matched exclusive-device efficiency comparisons. The original deadline remains unchanged.

## 2026-10-01 15:50 UTC — external GPU0 processes no longer observed

The read-only NVML snapshot at15:50:21.867UTC shows only this study's vLLM EngineCore658761 (parent final evaluator658477) onGPU0. External processes were not stopped by this agent. The exact end of external occupancy is not inferred from a sampled observation; snapshots are retained in results/shared_gpu_process_observations.jsonl and device history continues. Keep the frozen final evaluation settings (16GiB KV, memoryfraction.25, maxseqs256, final requestchunk1024) unchanged. Earlier shared-device elapsed times remain labeled accordingly; this observation does not make those earlier timings isolated measurements.

## 2026-10-01 16:43 UTC — figure presentation review after final freeze

Move the efficiency figure legend from individual axes to a shared external legend because the previous left-panel legend occluded several Frozen data points; increase figure height from5 to5.5 inches and reserve top margin. Plot values, method order, axes/log scaling, all measured timing/FLOPs inputs and all experimental protocols remain unchanged. The pre-final source snapshot remains immutable; only the live presentation script figures.py changes, with exact source diff and source hashes recorded in results/figure_layout_review.json. Regeneration completed and the resulting PNG was visually checked. Also reviewed gradient_drift and update_lr_sensitivity; the final report should state that update-LR panels have different y ranges. No new experiment, method selection, test scoring or metric definition change.


## 2026-10-01 19:36 UTC — resource forecast count correction

The inline remaining-time forecast refreshed during final evaluation grouped entries by exact `condition`, which excluded supplemental `*_mean_control` entries. The frozen 156-entry evaluation queue itself was complete and unchanged. The corrected resource-only forecast groups by model and task and asserts that planned, completed and remaining entries reconcile to all 156 versions. The prior affected snapshot is retained in `provenance/runtime_forecast_omitted_mean_matrix_20261001_1935.json`. No method, selection, scoring, evaluation protocol, run order or scheduling decision changed. Forecasts remain point estimates excluding startups and final CPU review.


### 2026-10-01T21:01:22.511378+00:00: final figure presentation review

After all 156 final versions completed, visual review found the final-accuracy legend overlapped ARC-Easy bars. Moved the legend outside the axes, increased figure height, and reserved top margin. The paired-interval title now explicitly says pointwise/unadjusted to distinguish the intervals from Holm-adjusted hypothesis tests. These are presentation-only changes in figures.py; values, ordering, color mapping, scoring, selected policies, adapters, frozen source snapshot and statistical calculations are unchanged. Existing efficiency legend correction remains. Final figures will regenerate automatically after checkpoint evaluation.


## 2026-10-02T04:01:53.978121+00:00 — final manual report and completion review

The interactive executor became unavailable after a managed-environment switch around2026-10-01 21:02UTC, reporting `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; attempted workspace writes also failed. No worker was restarted because of that observation failure. The existing pipeline independently completed all final/checkpoint generation and automatic analysis/report at21:29UTC. At2026-10-02 03:50UTC restored execution confirmed737complete versions/269646rows,156final versions/198030rows,52new checkpoint versions/104curve points, and terminated training/evaluation processes. The own resource monitor was stopped03:54UTC after verifying its command; external processes were untouched.

Added post-experiment CPU-only helpers finalize_narrative.py and completion_audit.py; they edit/report and audit retained artifacts, and never train, select candidates, regrade or change frozen sources. The autogenerated report is preserved underprovenance/report_automated_before_manual_review.md. Manual report adds main conclusions, exact selected configurations, all fixed seed scores, strong-baseline caveats, output-budget statistics, and future hypotheses. Nine PNGs were visually reviewed; PDF companions and all source hashes are retained. An initial audit-helper JSONL reader used str.splitlines, which splits Unicode separators inside valid strings; corrected it to physical file-line iteration before the audit passed. No data or experimental code was changed by that correction.
