# Results Index

[Project overview](../README.md) · [Research roadmap](EXPERIMENTS.md)

Each topic has one main report with its configuration, result tables, and limitations. This page provides navigation across the studies.

| Topic | Main report | Status / scope |
|---|---|---|
| Domain sensitivity and computational cost | [Accuracy, configurations and cost](../predictor/countdown_predictor_results/scaling-domains-20260930/README.md) | Latest; 13 conditions / 9 tasks, warmup and update LR sweeps; short-run Adaptive near Oracle, long-run Frozen degradation |
| Model and task scaling | [1.5B/7B results, configuration and attempts](../predictor/countdown_predictor_results/scaling-benchmarks-20260926/README.md) | Earlier study; three math tasks, 12 main conditions, 8 modified families; no general accuracy or compute advantage established |
| Predictor history transfer | [Matched carry/reset report and figure](../predictor/countdown_predictor_results/predictor-history-transfer/README.md) | 24 paired probes, fresh test1024, better gradient prediction without established net model improvement |
| Countdown one-step inputs and dynamic calibration | [Countdown report](../predictor/countdown_predictor_results/README.md) | Current; one-step selection, calibration, and completed independent/joint 3/8/19 comparisons |
| Earlier Countdown2 and lightweight calibration | [Historical comparisons](../predictor/countdown_predictor_results/HISTORY.md) | Completed reference experiments with different calibration settings |
| Early Countdown feasibility and feedback | [Local and feedback predictors](../predictor/predictor_countdown/README.md) | Earlier localized SFT, model-state transfer, and predicted-gradient updates |
| Static gradient prediction and data selection | [MATH/GSM8K results](DATA_SELECTION_RESULTS.md) | Earlier Qwen2.5-1.5B study |
| SFT stability and supervision | [Research archive](../research/README.md) | Published diagnosis and generalization studies |
| All-layer predictor updates | [Next experiments](EXPERIMENTS.md#next-experiments) | Deferred until the three-layer joint-update gap is understood; no completed all-layer accuracy claim |

Read comparisons within their recorded model, split, optimizer, and seed settings. Gradient cosine, downstream accuracy, and compute cost answer different questions.
