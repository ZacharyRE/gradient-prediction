# Post-freeze development: more inputs and model capacity

These exploratory results do **not** replace the original frozen 435-entry comparison. They use the same 513 development examples, starts 301/311/321 with update seeds 401/411/421, the full 4,097-example training pool and LR 0.001. Time is historical prefix preparation plus updates, excluding warmup, offline fitting and evaluation; these are not three standalone timing repetitions.

## Downstream results

| Fitting inputs / width | Update policy | Accuracy at 256 | Seconds at 256 | Accuracy at 512 | Seconds at 512 |
|---|---|---:|---:|---:|---:|
| 512 inputs, width 256 | Full CV16 | 82.65% | 20.77 | 83.11% | 37.48 |
| 512 inputs, width 256 | CV16 first 32, then Oracle16 | 83.17% | 18.61 | 83.24% | 33.52 |
| 2,048 inputs, width 256 | Full CV16 | 82.07% | 21.13 | 83.17% | 37.96 |
| 2,048 inputs, width 256 | CV16 first 32, then Oracle16 | 82.13% | 19.02 | 83.95% | 33.98 |
| 2,048 inputs, width 512 | Full CV16 | 82.39% | 21.36 | 80.90% | 38.51 |
| 2,048 inputs, width 512 | CV16 first 32, then Oracle16 | 82.59% | 19.25 | 83.11% | 34.40 |
| 2,048 inputs, width 512 | Full CV8 | 80.25% | 19.94 | 83.82% | 36.05 |

For comparison, the previously selected native Oracle32 reaches **83.95% at 256 updates / 27.43s** on these three development starts. Increasing fitting diversity changes the early-use 512-step mean by **+0.71pp** versus its 512-input control, but changes full-CV16 at 512 by only **+0.06pp**. At 256 the differences are **-1.04pp** for early use and **-0.58pp** for full CV16. Improving one long-horizon checkpoint does not establish a better time/quality frontier. The early-use recipe performs no predictor inference after step 32.

No new point passed both the 83.45% development-accuracy and 27.43s training-time thresholds. This gate was registered during late development, after some pilot readouts and before the complete three-start results. The conditional independent confirmation was not launched.

## Matched fitting exposure and actual cost

All three fits use the same 12 development teacher states. The 512-input/40-epoch and 2,048-input/10-epoch fits each visit **245,760 example-state pairs**. The 512 rows form an exact nested subset of the 2,048 rows; all original training shards and predictor-dev shards are byte-identical in the expanded cache. Optimizer-step counts differ slightly because full sequences are packed into batches. The original inherited p123 predictor already used 2,050 inputs; this ablation concerns the new state-aware architecture.

| Model | Epochs | Optimizer steps | Selected residual ratio | Label collection, s | Fit, s | Construction components, s |
|---|---:|---:|---:|---:|---:|---:|
| state256_coverage_fast | 40 | 15,840 | 0.9197 | 63.2 | 346.6 | 536.8 |
| state256_diversity2048 | 10 | 15,652 | 0.9145 | 215.9 | 395.6 | 738.4 |
| state512_diversity2048 | 10 | 15,652 | 0.9094 | 215.9 | 356.1 | 699.0 |

Construction components include the recorded label collection, complete fit, prerequisite teacher trajectories and source warmups. Shared dependencies are charged once per model; do not sum model rows. These are historical receipts, not clean cold-start measurements. Imports and unrecorded I/O remain additional. The fresh 512-input control at start 301 exactly reproduced all four original coverage checkpoints.

| Model | Oracle 256 residual ratio | Oracle 512 residual ratio |
|---|---:|---:|
| state256_coverage_fast | 0.9735 | 1.0013 |
| state256_diversity2048 | 0.9683 | 0.9957 |
| state512_diversity2048 | 0.9626 | 0.9936 |

Residual ratio compares centered prediction error variance with true-gradient variance on 129 predictor-development examples. These states also participate in fitting/selection. Values near 1 at 512 indicate little variance reduction on this diagnostic, not a theorem about Adam or downstream quality. Better selected gradient metrics did not consistently improve actual rollouts.

## Bounded token-distillation readout

The three previously registered KD fits use the original 8-state/512-input cache. Their readout is a single-start development screen at start 301, not an independent confirmation. Pure token-critic updates use a restricted answer-token objective; CV16 retains the original teacher correction objective.

| Recipe | Accuracy at 256 | Prep + updates, s |
|---|---:|---:|
| token_kd128_cv16_lr0.001 | 81.48% | 27.33 |
| token_kd128_nograd_cv16_lr0.001 | 81.48% | 27.14 |
| token_kd128_nograd_prediction0_lr0.0003 | 72.71% | 10.77 |
| token_kd128_nograd_prediction0_lr3e-05 | 74.85% | 11.20 |
| token_kd128_prediction0_lr0.0003 | 59.45% | 11.20 |
| token_kd128_prediction0_lr3e-05 | 65.89% | 11.31 |
| token_kd256_cv16_lr0.001 | 82.26% | 27.38 |
| token_kd256_prediction0_lr0.0003 | 73.10% | 11.65 |
| token_kd256_prediction0_lr3e-05 | 74.27% | 11.70 |

| Critic | Selected epoch / 50 | Complete fit, s | Recorded construction components, s |
|---|---:|---:|---:|
| token_kd128 | 3 / 50 | 852.7 | 972.1 |
| token_kd128_nograd | 2 / 50 | 295.1 | 414.5 |
| token_kd256 | 3 / 50 | 850.6 | 970.0 |

Construction charges existing source states, cached labels and the full fit; shared dependencies are deduplicated within each model, so rows must not be added together.

## Other token-critic configurations

Four previously fitted token critics were screened with two learning rates and three update policies at start 301. Each row below is the **best observed development accuracy among eight recipes**, selected after seeing these outcomes, with no independent confirmation. All 24 recipes and their 32/128/256-step readouts remain in the numerical inventory.

| Policy | Best observed recipe | Accuracy at 256 | Prep + updates, s | Accuracy range across eight recipes |
|---|---|---:|---:|---:|
| CV4 | token_random128_cv4_lr0.0003 | 80.70% | 27.87 | 72.51–80.70% |
| CV16 | token_value128_cv16_lr0.001 | 82.26% | 28.21 | 80.70–82.26% |
| Pure prediction | token_random128_prediction0_lr0.0003 | 76.80% | 11.64 | 67.45–76.80% |

Warmup accuracy at this start is 74.66%. Thus the best pure predictor shows a development improvement, but short Oracle32 reaches 80.12% in 4.71s at only 32 updates. Predictor construction remains additional; a positive change from warmup does not establish a time-to-quality advantage.

## Can the token critic predict the updated agent’s behavior?

This separate development diagnostic uses 513 full prompts, three starts and five existing Oracle states per start. These are the existing **gradient-selected checkpoints**, including the distillation-only fit; they were not selected for behavior accuracy. It tests whether the same predictor can directly substitute for the changed agent, not the best achievable behavior distillation. It compares forced-format ABC predictions with the current teacher and with simply retaining the warmup teacher’s predictions. The same answer template is supplied independently of the gold label; causal and label-invariance checks passed. It does not measure free-generation behavior.

| Critic | Oracle steps | Critic accuracy | Current teacher accuracy | Agreement with current teacher | Unchanged warmup agreement | Critic readout, s | Teacher readout, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| token_kd128 | 256 | 41.46% | 84.15% | 41.26% | 83.11% | 0.09 | 0.65 |
| token_kd128 | 512 | 42.24% | 83.37% | 38.53% | 82.33% | 0.09 | 0.65 |
| token_kd128_nograd | 256 | 45.16% | 84.15% | 44.96% | 83.11% | 0.09 | 0.65 |
| token_kd128_nograd | 512 | 45.81% | 83.37% | 44.90% | 82.33% | 0.09 | 0.65 |
| token_kd256 | 256 | 46.59% | 84.15% | 48.15% | 83.11% | 0.09 | 0.65 |
| token_kd256 | 512 | 45.94% | 83.37% | 45.42% | 82.33% | 0.09 | 0.65 |

Times are historical inference components averaged across starts; shared prefix preparation, model loading and offline fitting are additional. The first batch also includes the invariance checks. These development outcomes do not alter the frozen held-out comparison. [All 15 model/state summaries](results/token_behavior_development_summary.json).

## Completion of previously registered controls

These are single-start screens (301/401), not new held-out confirmations. “Needed” prepares frozen-prefix caches only for the fixed minibatch schedule, without consulting model outputs. “Class batched” packs class-stratified true-gradient samples together; it does not use a neural predictor. Times remain historical preparation/update components.

| Control | Updates | Accuracy | Prep + updates, s |
|---|---:|---:|---:|
| class_batched16_lr0.0003 | 256 | 82.46% | 19.58 |
| class_batched16_lr0.001 | 256 | 81.68% | 19.64 |
| class_batched4_lr0.0003 | 256 | 79.92% | 19.01 |
| class_batched4_lr0.001 | 256 | 83.24% | 19.01 |
| needed_class_cov16_s256 | 256 | 82.46% | 18.75 |
| needed_class_cov4_s256 | 256 | 80.70% | 16.65 |
| needed_cv16_s256 | 256 | 82.46% | 20.59 |
| needed_subset16_s256 | 256 | 82.65% | 18.31 |

The remaining minibatch gradient-memory grid uses candidate0 (1,024 training examples), start 301/401, and runs completed recipes through 1,024 updates. The following are **posthoc best development checkpoints among the completed recipes**, not independently selected or confirmed final methods. The budget cutoff left unequal learning-rate coverage; all completed checkpoint results and uncompleted recipe names remain in the inventories.

| Family | Best observed recipe | Updates | Accuracy | Prep + updates, s |
|---|---|---:|---:|---:|
| Oracle32 | oracle32_lr0.0003 | 128 | 81.09% | 13.76 |
| True-gradient subset | subset8_lr0.001 | 512 | 81.48% | 31.13 |
| Original CV | cv16_lr0.0003 | 256 | 80.51% | 18.18 |
| Gradient memory | table4_boot0_ema_lr0.0003 | 256 | 81.48% | 17.16 |

Registered optional recipes completed: **51**; incomplete or unexecuted: **12**. Completed checkpoints with a full development readout: **167**; still unscored: **0**. The additional token-critic and memory-table grids, when completed, are included in the linked numerical inventory. An unexecuted or partial recipe is not a measured failure.

[All late readouts, per-start differences and fitting receipts](results/late_development_summary.json). Other completed optional screens remain in the complete experiment inventory; missing readouts are not measured failures.

The [late-record audit](results/late_generated_records_audit.json) checked all **142,614 generated records across 278 readouts**, including complete source prompts and recomputed grading.
