# Literature and experimental implications

These studies suggest interventions, not local evidence that a recipe works. Our student is an already instruction-tuned Qwen2.5-1.5B; comparisons using a math base model, different evaluation temperatures, or much larger data cannot be imported as guarantees.

| Primary source | Evidence and relevance | Important limits |
| --- | --- | --- |
| Zhao et al., [Mind the Gap](https://arxiv.org/html/2509.15157v1), 2025 | Uses multiple student samples, guided re-solving for unsolved questions, then expert fallback. Target rewriting improves their SFT results. Motivates testing supervision beyond greedy-correct examples. | About 48k NuminaMath examples, 7B/8B models and 16 stochastic evaluation runs. Their weighting approximates its denominator as one; this does not make ordinary CE an unbiased final-answer-reward estimator. |
| Zelikman et al., [STaR](https://arxiv.org/abs/2203.14465), 2022 | Bootstraps verified rationales and uses answer-conditioned rationalization to cover failures. | Our initial experiment is one round of offline SFT, not full iterative STaR; answer-correct traces can still contain invalid reasoning. |
| Yuan et al., [Scaling Relationship on Learning Mathematical Reasoning](https://arxiv.org/abs/2308.01825), 2023 | Rejection-sampling fine-tuning uses generated correct solutions as supervision; motivates sampled rather than exclusively greedy targets. | Dataset/model scaling results do not establish improvement for this instruction-tuned checkpoint. |
| Bansal et al., [Smaller, Weaker, Yet Better](https://arxiv.org/html/2408.16737v2), 2024 | Compares synthetic solution coverage, diversity and false positives. More samples from a weaker generator can outperform fewer strong-generator examples under their compute budgets. | Our n8 student vs n1 teacher comparison is not FLOP matched; matched question/source controls are required for causal interpretation. |
| Gupta et al., [Selective Self-to-Supervised Fine-Tuning](https://arxiv.org/html/2502.08130v1), 2025 | Preserving correct model outputs motivates a retention control alongside new correct supervision. | Better preservation relative to raw SFT is distinct from positive gains over the untrained student. |
| Chu et al., [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161), 2025 | Studies generalization to rule/visual variants in GeneralPoints and navigation. | These tasks cannot prove SFT or LoRA universally incapable of generalization. |
| Lin et al., [Debunk the Myth of SFT Generalization](https://arxiv.org/html/2510.00237v1), 2025 | Shows that prompt diversity and CoT supervision can change SFT transfer results on decision-making tasks. | Task-limited counterevidence to a universal claim; not a demonstrated MATH repair. |
| [RL Fine-Tuning Heals OOD Forgetting in SFT](https://arxiv.org/abs/2509.12235), 2025 | Proposes tracking OOD forgetting across the SFT trajectory. | We reserve confirmation benchmarks rather than selecting checkpoints on their curves; reused data are explicitly exploratory. |
| Thinking Machines Lab, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), 2025 | Separately tunes rank/LR and emphasizes module coverage. | SFT log-loss comparisons do not guarantee mathematical final-answer accuracy. Prior local module/rank controls already failed on raw targets. |

Downloaded original PDFs and SHA256 hashes are stored in `download_manifest.json`. Dated entries below retain their original prospective wording; the subsequent completed local experiments are documented in `../CAUSAL_FINDINGS.md` and the final report. In particular, the complete DFT/KL coefficient grid and cross-source DFT controls have now finished; their earlier queued status is historical.

The [Qwen2.5 Technical Report](https://arxiv.org/html/2412.15115v2), sections 4.1–4.3, describes over one million SFT examples, mathematical CoT generated with rejection sampling/reward guidance, and subsequent preference/RL stages. Our exact checkpoint is already Instruct; additional 1–2k-example SFT is an incremental adaptation experiment. This motivates testing genuinely new supervision and retention, but does not establish that the local model is saturated or that overlap with its undisclosed post-training data caused the observed regression. Qwen2.5-Math-1.5B base-model results cannot substitute for this fixed Qwen2.5-1.5B-Instruct baseline.

[Robust fine-tuning of zero-shot models (WiSE-FT)](https://arxiv.org/abs/2109.01903), Wortsman et al., CVPR2022, motivates a bounded diagnostic: interpolate the frozen base and SFT weights. For ordinary LoRA without other updated parameters this is implemented by scaling alpha, leaving A/B unchanged. The paper studies CLIP and related vision transfer, so it supplies an intervention idea, not evidence that this LLM will improve. Our scale is selected on reused development only and must be frozen for confirmation.

[The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301), Zhang et al.,2025, explicitly distinguishes correct final answers from valid processes and documents evaluation biases from rewarding flawed reasoning that reaches the correct answer. This supports checking intermediate steps, not assuming a PRM/LLM judge is infallible. Local counterexamples for targets1441/1461/208 are independently evaluated with rational/binomial/symbolic arithmetic in audits/process_counterexamples.json. No PRM has been used in our training or filtering.

## 更贴近本实验的外部结果（17:11 UTC 补查）

- [SmartAD, ACL Findings2026](https://aclanthology.org/2026.findings-acl.1349.pdf)：官方最终版Table2的1.5B **CoT Prompt48.2、CoT Distill44.2、SmartAD44.4**（Math500）。因此其“总体优于baseline”不能读成纯数学CoT SFT超过原基座；多跳QA与工具能力带来不同收益。使用2000MATH+1000HotpotQA、Qwen2.5-32B教师、LoRA2epochs，附录LR2e-4、batch4、maxlen10240；非本地同协议复现或多seed证明。可借鉴同题候选按学生NLL筛选的思路，不能把agent/tool结果当成本任务修复证据。PDF及文本已保存smartad_acl2026.*。
- 搜到的 [Qwen1.5B LoRA R1模型卡](https://huggingface.co/PursuitOfDataScience/Qwen2.5-1.5B-Instruct-Lora-Deepseek-R1) 把“16次中至少一次正确”称作Pass@1，且无匹配原基座对照；不作为有效性证据。
- [Plasticity vs Rigidity, EACL2026](https://aclanthology.org/2026.eacl-srw.37/) 是RLVR，非SFT。rank256结果不能替代本任务SFT实验证据。
- [JordanAsh实验仓库](https://github.com/JordanAsh/plasticity-rl-experiments) 使用Qwen2.5-1.5B **base** 和RL过程中正确轨迹再SFT，非目前Instruct+LoRA的直接对照。
- [公开rank-demand实验的旧简报](https://raw.githubusercontent.com/cxcscmu/lora-rank-demand/main/docs/ADVISOR_BRIEF_2026-08-11.md) 顶部明确标记已被[第二学习率结果](https://raw.githubusercontent.com/cxcscmu/lora-rank-demand/main/docs/TWO_RATE_FINDINGS_2026-08-16.md)取代，原正结果未复现。未把其旧rank/几何相关性当结论，也未验证其整个训练实现。此处只记录检索排除理由。

### OpenMathInstruct-2（新增，2026-09-09）

来源：https://arxiv.org/abs/2410.01560 ，最终v2 PDF与文本已保存；数据固定revision469216e3f46f4dacf476b382e192485ea51a143e。关键限制：论文训练的是Llama3.1-8B-Base，不是对同一个Instruct checkpoint继续LoRA SFT，所以67.8对51.9不能作为本地设定的正复现。匹配问题/每题解答数的格式对照中，较短OpenMath CoT优于verbose Llama CoT；问题多样性与数据规模提供独立假设。Table3中三种过程/奖励过滤没有产生明显收益，反驳“发现错误过程就足以解释SFT无收益”的过度推断。我们只把这些作为需要本地对照的假设。数据中合成题expected_answer来自majority vote；不能当作独立人工真值。


## Qwen2.5-Math-7B-Instruct official prompt check, 2026-09-09

Pinned modelcard ef9926d75ab1d54532f6a30dd5e760355eb9aa4d savedas qwen_math7_model_card.md (SHA9cdd5eeb...). OfficialpureCoT putsstep-by-step/boxed instructioninSYSTEM andtheprobleminUSER, distinctfromTIRprogram-useprompt. Beforeanymathteachergeneration, adoptedthisofficialCoTprompt; studenttraining/evaluationpromptunchanged. Thisis a combinedteacher+prompttargetpipelinecomparison, not a checkpoint-onlycausaltest. Modelcard distinguishesInstructchatfromMathBasefine-tuningstartingpoint; do notsubstitutestudentorusepublishedTIRscoreasourCoTresult. Primarysource: https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct .


## Objective controls: DFT and anchoring (2026-09-09)

[DFT v3](https://arxiv.org/html/2508.05629v3), Appendix A.6/Table9, explicitly tests LoRA r8/alpha16: Qwen2.5-Math-1.5B base MATH50031.66, SFT41.47, DFT64.85; five-benchmark macro15.92/16.87/32.90. It uses100k NuminaMath, base checkpoints,16 stochastic decodes at4096, so this is neither our Instruct baseline nor multi-training-seed replication. Local prior raw-target DFT failed; new matched student/32B-target objective controls now test the unmeasured interaction. Implementation uses detached token probability times CE. The inverse-probability RL interpretation is not evidence that ordinary softmax-logit CE gradients explode locally: dCE/dz=p-onehot is bounded. Weighting changes token directions and Adam trajectories, not just scalar LR.

[Anchored SFT v3](https://arxiv.org/html/2509.23753v3), Eq6, adds forward KL(base||adapted) to DFT. Math experiments use Qwen2.5-7B,10k–100k NuminaMath and16 stochastic4096 decodes. Its explicit LoRA result is medical LLaMA2-7B, not local1.5B-Instruct mathematics. It motivates a retention hypothesis. Local CE+forwardKL0.1/1 now completed on the same1547 general7B targets: dev301/309 versus plainCE311, with no demonstrated retention repair. This is not fullASFT. Matched DFT+KL0/0.1/1 is prospectively queued;0.1 was added before any teacherDFT training/scores to avoid relying on one strong coefficient. The paper's approximately0.1 optimum is from medical QA and is not a known optimum for this mathematics task. We do not import its theoretical generalization claims as measured local causes.

## 2026-09-10 03:04 UTC：Minerva/OCW数值评分复核

[Minerva原论文附录F.5](https://arxiv.org/html/2206.14858v2)明确区分191numeric与81symbolic问题；提供数值单位移除、浮点/SymPy转换与一个特殊近零比较分支。本文不是通用数学MATH任务的另一个名字：这里的math-ai/minervamath是272道OCW定量STEM题，必须与lm-eval名为minerva_math的MATH提示配置区分。

[Qwen2.5-Math官方numeric_equal](https://github.com/QwenLM/Qwen2.5-Math/blob/615096eceb14e653ba791d862a21b326a1d26e12/evaluation/grader.py)使用相对1e-4的math.isclose；函数本身默认绝对容差0。本轮采用这一数值比较约定，但不声称完整复制Qwen的所有提取/百分比处理或原Minerva近零分支。

仅检查gold与自身等价不足以发现语义解析错误。本地58条e-notation参考被LaTeX解析为欧拉常数表达式；独立反例还发现10^-19与2×10^-19会被旧verify判等。191道新数值分支以安全科学计数法、有限实数和rtol1e-4/atol0处理，188字面数值+3明确表达式（np.arcsin(10/13)、-1./3、-3./2）全部覆盖。81符号题保留既有原始/严格方法；所有规则、1%/5%容差诊断及旧评分敏感性在任何OOD输出前登记。原题、参考字符串、模型输出及旧评分都不覆盖。

来源存档及sha见[minerva_scoring/sources.json](minerva_scoring/sources.json)。本轮发现不代表原MATH退化由它造成；额外MATH负例审计仅在4387个可识别实数gold中发现index2854的极小阶乘分数问题，开发435个为0，另以前置敏感性处理。


## 2026-09-10 09:33 UTC：外部正结果与尚未测试的token干预

新增[补查记录](late_crosscheck/NOTES.md)核对CFT论文的实际LoRA表格及两个同1.5B-Instruct公开结果的评测限制。没有将单seed/100题、不同提取器的结果当成本地稳定提升。CFT未在本轮实现，仅保留为未验证假设；原有full/dense对照直接复用，不能外推到未测试的生成目标/full组合。
