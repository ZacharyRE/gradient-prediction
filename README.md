# Gradient Prediction for LoRA Learning

Can a learned predictor estimate useful gradients and keep improving a model as its parameters change?

**Current focus:** predict LoRA gradients at lower cost while preserving downstream learning, then test whether the predictor can be reused across starting models and training datasets as a training surrogate.

**Latest — reusable training surrogates (October 15):** early predictor use approaches Oracle accuracy, but no overall efficiency or data-selection advantage is established. Tests three new starting adapters, random and source-disjoint genre datasets, behavior prediction, matched/tuned/shorter Oracle controls and complete time/cost accounting. [Study, evidence and reproduction →](reports/predictor_surrogate_20261015/README.md)

**Previous — nine-hour predictor efficiency study (October 14):** CV4 still misses the quality target. Online RLS8 approaches Oracle on MNLI without offline fitting, but early-stopped Oracle remains faster. A secondary MNLI CV16 result reaches **84.68% vs early Oracle's 84.97%**, using **23.20 vs 28.93 seconds** of adaptation with an already-fitted predictor; offline preparation is additional. PAWS's primary CV16 falls short. Includes 48 verified timing replays, FLOPs and long-run diagnostics. [Results and conclusions →](reports/predictor_efficiency_20261014/README.md)

**[Start here — consolidated 0.5B / 1.5B / 7B experiment overview](reports/experiment_overview_20261013/README.md)** (October 13): all evaluated task coverage, Base / Warmup / Oracle / predictor scores, exact settings, method comparisons, FLOPs and measured time. **Gradients are learnable; matched-quality training speedup remains unproven.** Includes one overview figure and downloadable evidence tables.

**New — 1.5B across 9 benchmarks:** oracle gains are **+1.95–7.03 pp** across all nine tasks. CV4 retains **68% on MNLI / 57% on PAWS**; other predictor gains remain uncertain, and an advantage over true-gradient controls is unproven. [Results, figure and exact settings →](reports/benchmark_1p5b_20261011/README.md)

**Previous — 0.5B across 13 benchmarks:** oracle gains reach **+8.77 pp on BoolQ** and **+13.18 pp on PAWS**; predictors retain **+3.50 / +7.01 pp** using four true-gradient corrections per step. Seven of ten admitted configurations have positive final mean gains for both methods; a compute advantage remains unproven. [Results, figure and configurations →](reports/benchmark_0p5b_20261010/README.md)

**Oracle improvement follow-up — four tasks:** single-layer true-answer updates establish gains on BoolQ and OpenBookQA. ARC-Easy also improves, partly through formatting; ARC-Challenge remains uncertain. Teacher targets and extra layers show no established added benefit. [Six studies, confirmation runs and controlled comparisons →](reports/oracle_improvements_20261007/README.md)

**Latest BoolQ result — 1.5B predictor data scaling:** Oracle improves **78.99% → 81.70%**, but Adaptive reaches only **75.10–76.12%** with 2K–6K offline examples. Initial gradient prediction improves with more data, then degrades sharply during updates. [Results, one-step comparison and diagnostics →](countdown_predictor_results/boolq-predictor-datascale-20261010/README.md)

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
| 9. Time to downstream quality | Can a gradient surrogate approach a tuned Oracle faster, beyond 32 updates? | [Nine-hour 1.5B PAWS/MNLI study](reports/predictor_efficiency_20261014/README.md): calibration, long trajectories, online linear surrogates, frozen readouts and complete cost definitions |
| 10. Reusable training surrogates | Can one fitted predictor transfer to new starting adapters/data, select useful data, and predict changed behavior? | [Frozen 1.5B/MNLI transfer and selection study](reports/predictor_surrogate_20261015/README.md): separate direct-training and data-selection endpoints, repeated timing and explicit offline costs |

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

Evaluation preserves **all input tokens**. A generation-token cap limits newly generated output only. Calibration uses reference solutions for gradient supervision; downstream accuracy generation receives question prompts. Cost conclusions are specific to each report's tasks, implementation and preparation accounting. True-gradient corrections still require backpropagation.
