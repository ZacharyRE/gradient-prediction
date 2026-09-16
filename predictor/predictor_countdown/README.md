# Countdown: from localized LoRA SFT to predicted-gradient updates

**Research question:** Can a learned gradient predictor drive useful LoRA updates at a single linear module of Qwen2.5-0.5B-Instruct?

The evidence supports three distinct steps: localized SFT works; gradients are predictable at a fixed model state; output-error-conditioned predictions can improve held-out accuracy during subsequent updates. Accurate gradient prediction alone does not establish effective optimization.

## 1. Establish that localized SFT can learn the task

An earlier experiment trained on 4,096 solver-generated Countdown solutions, with 1,024 development and 2,048 held-out test questions. Only LoRA parameters were trained.

| Scope | LoRA parameters | Test accuracy |
|---|---:|---:|
| Base model | 0 | 4.20% |
| All seven linear projections across all 24 blocks | 35,192,832 | 30.03% |
| All seven linear projections in block 8 only | 1,466,368 | 23.52% |

The trained results are means over seeds 101/102/103. Layer indices are zero-based. This establishes single-block adaptation on this task, not that every individual projection works equally well.

The subsequent predictor experiment narrowed adaptation to **block 8 `self_attn.o_proj` alone**, with rank=alpha=64 and **114,688** LoRA parameters. Its real-gradient AdamW reference reached **14.06%**, versus **4.49%** at the shared 32-step warmup checkpoint. This supplies single-module feasibility evidence within the predictor experiment. The earlier block-level study and this experiment use different data/update budgets and are not a controlled comparison of localization alone.

## 2. Learn to predict the single-module gradient

Offline, run real backpropagation on training questions and their complete reference solutions. Record local activations and the gradient of summed completion cross-entropy with respect to the `o_proj` output. Train a predictor to estimate this activation gradient, then analytically convert it into full LoRA A/B parameter gradients.

**Local route.** A tokenwise MLP reads the module input, block output, supervision mask and position. A conditional MLP additionally reads frozen correct-next-token embeddings and a mean solution embedding. Bidirectional Transformer variants allow the predictor itself to exchange information across the full sequence. Deployment needs a forward pass through block 8, without the upper model's forward or backward pass.

**Feedback route.** Run the complete model forward and analytically compute the final residual's error from logits, training labels, readout and RMSNorm. A learned network approximates transporting that error back to block 8. It performs no original-model autograd backward during deployment, but does require the full forward and analytic output derivatives. Zero output error implies zero predicted activation gradient.

### Fixed-state gradient accuracy

All metrics below use 256 held-out questions at the same step-32 LoRA state. Cosine and relative L2 refer to the **complete 114,688-dimensional A+B gradient**, averaged per question, not a compressed projection or token activation gradient.

| Predictor | Training questions | Cosine ↑ | Relative L2 ↓ |
|---|---:|---:|---:|
| Local MLP | 1,536 | 0.563 | 0.813 |
| Label-conditioned MLP | 1,536 | 0.601 | 0.783 |
| Bidirectional Transformer, width 512 × 2 layers | 1,536 | 0.730 | 0.682 |
| Main local Transformer, same architecture | 8,192 | 0.786 | 0.605 |
| Larger local Transformer, width 768 × 4 layers | 8,192 | 0.798 | 0.589 |
| Feedback large | 8,192 at state 32; 1,536 at state 96 | 0.850 | 0.504 |
| Feedback with replay | Same question pool; additional training states | 0.854 | 0.497 |

The main local model's cosine falls from **0.786 to 0.370** when moved to state 96. At previously unseen intermediate state 64, it scores **0.478**, a local model trained at states 32/96 scores **0.606**, and feedback large scores **0.717**. Feedback large scores **0.780** at state 96, which was included in its training state coverage. These results distinguish generalization to new questions from generalization to changing LoRA states; the route comparison also changes information access and training coverage.

## 3. Freeze the predictor and test actual learning

Use a separate 1,536-question update set with reference solutions to update LoRA from the shared step-32 checkpoint. Select candidate checkpoints on development accuracy, freeze selection, then evaluate all 2,048 test questions. Test inference receives questions only and does not update the model.

| Method | Correct / 2,048 | Accuracy |
|---|---:|---:|
| Base | 86 | 4.20% |
| Shared real-gradient warmup, 32 steps | 92 | 4.49% |
| Local multi-state predictor, SGD 16 steps | 92 | 4.49% |
| Fixed mean gradient, SGD 16 steps | 97 | 4.74% |
| Real gradient, SGD 16 steps | 102 | 4.98% |
| **Feedback large, SGD 384 steps** | **127** | **6.20%** |
| Fixed mean gradient, SGD 384 steps | 3 | 0.15% |
| Weight decay only, 384 steps | 91 | 4.44% |
| Real gradient, SGD 384 steps | 143 | 6.98% |
| Local predictor + learned rank-4 update preconditioner | 269 | 13.13% |
| Fixed mean gradient + same-sized learned preconditioner | 261 | 12.74% |
| Real gradient, AdamW 192 steps (different optimizer setting) | 288 | 14.06% |

Feedback improves over warmup by **1.71 percentage points**, paired 95% CI **[0.54, 2.88]**, McNemar p=0.00548. Its difference from matched real-gradient SGD is −0.78 points, CI [−1.76, 0.15]; this does not establish equivalence.

The local SGD route shows no accuracy gain. The learned-preconditioner combination improves accuracy substantially, but exceeds the fixed-mean combination by only 0.39 points, CI [−0.93, 1.71]. Its gains cannot be attributed specifically to example-conditioned gradient prediction.

## Interpretation and limits

- **Supported:** task-specific localized adaptation, useful fixed-state gradient prediction, and a positive feedback-driven update result.
- **Main obstacle:** prediction errors under changing LoRA states; high static cosine does not guarantee stable optimization. Longer rollouts can deteriorate.
- Predictor fitting and the 32-step warmup use real gradients. This is not end-to-end backpropagation-free training from initialization.
- The predictors use extra gradient-supervised questions and have more parameters than the LoRA itself. No total-data, FLOP, memory or wall-clock advantage has been established.
- Predictor experiments use one training seed. Their 10,000-resample paired bootstrap intervals describe question uncertainty, not training-seed uncertainty; comparisons are exploratory and not multiplicity-adjusted.
- Conclusions concern this Countdown distribution and chosen module. Earlier SFT was sensitive to prompt style and reduced out-of-task math performance.
- Evaluation retains every input token. **2,048 is the maximum number of newly generated output tokens**, not an input limit. The predictor experiment uses greedy native-vLLM BF16 inference, seed 42; earlier experiments must be interpreted under their own recorded inference settings.

## Compact release

- [TRAINING.md](TRAINING.md): architecture, objectives, data roles and effective training settings.
- [configs.json](configs.json): selected original run configurations, with machine paths normalized.
- [results.json](results.json): numerical gradient, accuracy and paired statistical summaries.
- [selection.json](selection.json): original frozen selection records and split identity metadata.
- [sources.json](sources.json): SHA256 provenance of the local source artifacts.

This is a configuration and results release, not a standalone runnable training package. Weights, activation caches, datasets and per-question generations are omitted to keep it small. Original evaluation inputs remain intact in the local archive; none were shortened for this release.
