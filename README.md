# Gradient Prediction for LoRA Learning

Can a learned predictor estimate useful gradients and keep improving a model as its parameters change?

**Current focus:** use a bidirectional Transformer to predict activation gradients, reconstruct LoRA A/B gradients, and maintain useful updates through online calibration across model sizes and math tasks.

**Latest follow-up — efficient 0.5B updates:** a smaller predictor with four true-gradient probes reaches **12.89% vs Oracle's 23.54%** on the original Countdown test. Estimated optimization FLOPs are lower, but matched-quality efficiency and GPU wall-time savings remain unverified. [Setup, results and limitations →](predictor/countdown_predictor_results/efficient-test-20261002/README.md)

**Cross-domain results — broader domains and measured cost:** across 13 model/task conditions, Adaptive is close to Oracle, but adds only **+0.10 pp** over warmup on average (Oracle: **+0.02 pp**). Frozen is **4.1–5.9× faster per online step**; Adaptive uses fewer counted FLOPs but takes **2.18–2.97×** the Oracle time. Four long-run Frozen tests deteriorate. [Results, settings and figures →](predictor/countdown_predictor_results/scaling-domains-20260930/README.md)

**Earlier scaling result — 1.5B / 7B:** across Countdown, GSM8K and MATH, activation gradients remain learnable and calibration prevents some frozen-predictor failures. However, all six gentle-warmup three-seed Adaptive-versus-Mean accuracy intervals include zero. No general downstream or compute advantage is established. [Results, configuration and attempts →](predictor/countdown_predictor_results/scaling-benchmarks-20260926/README.md)

**Predictor history transfer:** keeping past calibration weights reduces held-out gradient error in **24/24 paired probes**. On a fresh 1,024-question one-update evaluation, carry-history reaches **6.93% vs reset's 6.12%**, but remains near the **7.00%** unchanged-model baseline. [History-transfer report →](predictor/countdown_predictor_results/predictor-history-transfer/README.md)

**Previous milestone:** layers **3, 8, 19** have been tested both independently and jointly. Joint adaptive updates reach **8.04%**, below their **8.64%** warmup baseline; oracle reaches **10.73%**. Independent layer 8 remains close to oracle (**7.32% vs 7.45%**), while layer 3 is less stable. These are three-seed means on the earlier 2,048-question test set. [Layer comparison →](predictor/countdown_predictor_results/README.md#joint-and-independent-layer-updates)

## Research route

| Stage | Question | Evidence and reading |
|---|---|---|
| 1. Predictability | Do hidden states contain useful gradient information? | [Earlier MATH/GSM8K gradient prediction and data selection](docs/DATA_SELECTION_RESULTS.md) — Qwen2.5-1.5B |
| 2. Useful updates | Can predicted gradients drive actual learning? | [Early Countdown local and feedback predictors](predictor/predictor_countdown/README.md) — Qwen2.5-0.5B |
| 3. One-step selection | Which predictor inputs work best downstream? | [Y / mask / position ablation](predictor/countdown_predictor_results/README.md#one-step-input-selection) |
| 4. Dynamic calibration | Can small calibration batches sustain multi-step updates? | [4 calibration + 16 validation examples; 16/32-step results](predictor/countdown_predictor_results/README.md#dynamic-calibration) |
| 5. Layer scope | Does the single-layer method transfer to joint updates? | [Completed independent 3/8/19 and joint 3+8+19 comparison](predictor/countdown_predictor_results/README.md#joint-and-independent-layer-updates); all-layer expansion deferred |
| 6. History transfer | Does retaining predictor experience help on new model states and data? | [Matched carry/reset comparison](predictor/countdown_predictor_results/predictor-history-transfer/README.md): better gradient prediction, limited downstream benefit |
| 7. Model and task scaling | Does the method remain useful at 1.5B/7B on Countdown, GSM8K and MATH? | [Scaling results and tested changes](predictor/countdown_predictor_results/scaling-benchmarks-20260926/README.md): learnable gradients without established general accuracy gains |
| 8. Domain sensitivity and cost | Do predicted updates improve on warmup and reduce total cost? | [13 conditions, warmup/update LR sweeps, controlled profiling and 1,024-step follow-up](predictor/countdown_predictor_results/scaling-domains-20260930/README.md): limited extra accuracy; no established improvement at lower total cost |

The [SFT diagnosis studies](research/README.md) provide supporting evidence about training stability, supervision targets, and evaluation. Their settings differ from the Countdown predictor experiments.

## Key links

| Entry | Purpose |
|---|---|
| [Predictor research map](predictor/README.md) | Current topic and earlier predictor studies |
| [Countdown report](predictor/countdown_predictor_results/README.md) | Main results, method, configuration, and evidence links |
| [Results index](docs/RESULTS.md) | One entry point for results across research topics |
| [Research roadmap](docs/EXPERIMENTS.md) | Completed work, open questions, and next experiments |
| [SFT research archive](research/README.md) | Published reports, methods, and supporting evidence |

## Code and reproduction

[training/](training/), [evaluation/](evaluation/), [gradient_geometry/](gradient_geometry/), and [configs/gradient_geometry/](configs/gradient_geometry/) contain the shared implementation of the earlier gradient-prediction/data-selection pipeline. Start with its [setup and reproduction guide](predictor/single_layer/README.md#setup).

Countdown releases include [source snapshots](predictor/countdown_predictor_results/layer-comparison-32step/source/README.md), configurations, and compact results. They require the original data, weights, cached supervision, and experiment layout to rerun. Each report states its own reproduction boundary.

Evaluation preserves **all input tokens**. A generation-token cap limits newly generated output only. Calibration uses reference solutions for gradient supervision; downstream accuracy generation receives question prompts. No overall compute saving or backpropagation-free training claim is established.
