# Predictor Research Map

[Project overview](../README.md) · [Results index](../docs/RESULTS.md) · [Current roadmap](../docs/EXPERIMENTS.md)

The current research path is **gradient prediction → useful one-step updates → calibration under model changes → all-layer prediction**.

## Start here: Countdown

**[Main report: one-step input selection and dynamic calibration](countdown_predictor_results/README.md)**

The current predictor reads module output Y, supervision mask, and position. A bidirectional Transformer predicts activation gradients, which are converted analytically to LoRA A/B gradients. The completed experiments update layer 8 `o_proj` of Qwen2.5-0.5B-Instruct.

| Topic | Main question | Where to read |
|---|---|---|
| One-step input selection | Which inputs produce useful updates? | [Input ablation](countdown_predictor_results/README.md#one-step-input-selection) |
| Dynamic calibration | How do 4 current-batch calibration examples and 16 validation examples support 32 updates? | [Current method and results](countdown_predictor_results/README.md#dynamic-calibration) |
| Earlier calibration | What worked or failed in Countdown2 and the lightweight variant? | [Historical comparisons](countdown_predictor_results/HISTORY.md) |
| All-layer extension | Shared predictor or one predictor per layer; do downstream updates remain useful? | [Pending work](../docs/EXPERIMENTS.md#next-experiments) |

## Earlier studies

| Directory | Role | Model / task |
|---|---|---|
| [predictor_countdown/](predictor_countdown/README.md) | Early localized SFT, local predictors, feedback, and learned preconditioning; separate protocols | Qwen2.5-0.5B / Countdown |
| [single_layer/](single_layer/README.md) | Static raw-gradient prediction and data selection; original shared-code entry | Qwen2.5-1.5B / MATH and GSM8K |

Dataset sizes, input access, optimizers, and seeds are study-specific. Use each report's configuration when comparing results; the MATH/GSM8K conventions do not apply to every predictor here.

## Documentation roles

The Countdown topic README is the main maintained report. Its `adaptive-calibration-32step/` subdirectory stores the existing configuration and evidence bundle; its README is an artifact index. The historical report preserves prior comparisons. Future variants should extend the topic's report and evidence tables, with a separate topic only when the research question changes.
