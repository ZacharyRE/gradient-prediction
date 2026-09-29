# Predictor Research Map

[Project overview](../README.md) · [Results index](../docs/RESULTS.md) · [Current roadmap](../docs/EXPERIMENTS.md)

The current research path is **gradient prediction → useful one-step updates → calibration under model changes → history transfer and broader layer scope**.

## Start here: Countdown

**[Main report: gradient prediction, calibration, and layer updates](countdown_predictor_results/README.md)**

The current predictor reads module output Y, supervision mask, and position. A bidirectional Transformer predicts activation gradients, which are converted analytically to LoRA A/B gradients. Completed experiments cover independent layer 3, 8, and 19 updates and joint layer 3+8+19 updates of Qwen2.5-0.5B-Instruct.

| Topic | Main question | Where to read |
|---|---|---|
| History versus ordinary training | Does sequential history beat training on the same labels and budget? | [Ben learning audit: setup, results and cost](countdown_predictor_results/ben-learning-audit-20260927/README.md) |
| Model and task scaling | Does calibration help 1.5B/7B across Countdown, GSM8K and MATH? | [Results, configuration and attempts](countdown_predictor_results/scaling-benchmarks-20260926/README.md) |
| Predictor history transfer | Does retaining past calibration help under matched current model states and data? | [Matched results and figure](countdown_predictor_results/predictor-history-transfer/README.md) |
| One-step input selection | Which inputs produce useful updates? | [Input ablation](countdown_predictor_results/README.md#one-step-input-selection) |
| Dynamic calibration | How do 4 current-batch calibration examples and 16 validation examples support 32 updates? | [Current method and results](countdown_predictor_results/README.md#dynamic-calibration) |
| Earlier calibration | What worked or failed in Countdown2 and the lightweight variant? | [Historical comparisons](countdown_predictor_results/HISTORY.md) |
| Layer scope | How do independent 3/8/19 updates compare with joint 3+8+19 updates? | [Completed comparison](countdown_predictor_results/README.md#joint-and-independent-layer-updates) |
| All-layer extension | Can joint predictor updates improve on their own warmup baseline? | [Deferred pending diagnosis](../docs/EXPERIMENTS.md#next-experiments) |

## Earlier studies

| Directory | Role | Model / task |
|---|---|---|
| [predictor_countdown/](predictor_countdown/README.md) | Early localized SFT, local predictors, feedback, and learned preconditioning; separate protocols | Qwen2.5-0.5B / Countdown |
| [single_layer/](single_layer/README.md) | Static raw-gradient prediction and data selection; original shared-code entry | Qwen2.5-1.5B / MATH and GSM8K |

Dataset sizes, input access, optimizers, and seeds are study-specific. Use each report's configuration when comparing results; the MATH/GSM8K conventions do not apply to every predictor here.

## Documentation roles

The Countdown topic README is the main maintained report. The `adaptive-calibration-32step/` bundle preserves the original layer-8 study; `layer-comparison-32step/` adds joint and independent layer comparisons; `predictor-history-transfer/` isolates retained predictor experience. Bundle READMEs index configurations and evidence. The historical report preserves prior comparisons. Future variants should extend the topic's report and evidence tables, with a separate topic only when the research question changes.
