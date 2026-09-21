# Gradient Prediction for LoRA Learning

Can a learned predictor estimate useful gradients and keep improving a model as its parameters change?

**Current focus:** use a bidirectional Transformer to predict activation gradients, reconstruct LoRA A/B gradients, and maintain useful updates through online calibration on Countdown.

**Latest completed result:** single-layer calibration reaches **7.32%** test accuracy after **32 updates**, from **4.49%** before updating; the true-gradient oracle reaches **7.45%**. These are means over three paired seeds on the complete 2,048-question test set. Fixed-5 calibration is stronger at step 16; all-layer downstream results are still pending. [Read the main Countdown report →](predictor/countdown_predictor_results/README.md)

## Research route

| Stage | Question | Evidence and reading |
|---|---|---|
| 1. Predictability | Do hidden states contain useful gradient information? | [Earlier MATH/GSM8K gradient prediction and data selection](docs/DATA_SELECTION_RESULTS.md) — Qwen2.5-1.5B |
| 2. Useful updates | Can predicted gradients drive actual learning? | [Early Countdown local and feedback predictors](predictor/predictor_countdown/README.md) — Qwen2.5-0.5B |
| 3. One-step selection | Which predictor inputs work best downstream? | [Y / mask / position ablation](predictor/countdown_predictor_results/README.md#one-step-input-selection) |
| 4. Dynamic calibration | Can small calibration batches sustain multi-step updates? | [4 calibration + 16 validation examples; 16/32-step results](predictor/countdown_predictor_results/README.md#dynamic-calibration) — current main experiment |
| 5. All-layer prediction | Can the approach extend to all 24 o_proj modules? | [Current roadmap](docs/EXPERIMENTS.md#next-experiments) — downstream evaluation unfinished |

The [SFT diagnosis studies](research/README.md) provide supporting evidence about training stability, supervision targets, and evaluation. Their settings differ from the Countdown predictor experiments.

## Find what you need

| Entry | Purpose |
|---|---|
| [Predictor research map](predictor/README.md) | Current topic and earlier predictor studies |
| [Countdown report](predictor/countdown_predictor_results/README.md) | Main results, method, configuration, and evidence links |
| [Results index](docs/RESULTS.md) | One entry point for results across research topics |
| [Research roadmap](docs/EXPERIMENTS.md) | Completed work, open questions, and next experiments |
| [SFT research archive](research/README.md) | Published reports, methods, and supporting evidence |

## Code and reproduction

[training/](training/), [evaluation/](evaluation/), [gradient_geometry/](gradient_geometry/), and [configs/gradient_geometry/](configs/gradient_geometry/) contain the shared implementation of the earlier gradient-prediction/data-selection pipeline. Start with its [setup and reproduction guide](predictor/single_layer/README.md#setup).

Countdown releases include [source snapshots](predictor/countdown_predictor_results/adaptive-calibration-32step/source/README.md), configurations, and compact results. They require the original data, weights, cached supervision, and experiment layout to rerun. Each report states its own reproduction boundary.

Evaluation preserves **all input tokens**. A generation-token cap limits newly generated output only. Calibration uses reference solutions for gradient supervision; downstream accuracy generation receives question prompts. No overall compute saving or backpropagation-free training claim is established.
