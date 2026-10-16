# Frozen results

Qwen2.5-1.5B-Instruct; MNLI training-population source-group holdouts. Three new warmup starts 302/303/304. All horizons count updates after 32 warmup updates. Evaluation inputs are complete; 256 is the maximum number of newly generated output tokens.

Recipes were fixed before selection-dev outcomes. Candidate choices were then fixed before selection-holdout outcomes. Intervals resample starts and source groups and are descriptive with only three starts. A mean within 1 percentage point of Oracle is not a statistical noninferiority result.

The common 32-update target warmup averaged **4.23s** of updates and **10.27s** under its recorded setup/update/save timer. These are one historical run per start, not three timing repetitions. Warmup is shared and excluded from the incremental costs below.

## Full 4,097-example adaptation pool

| Method | Updates | Holdout accuracy %, 95% CI | Gap to population Oracle, pp, 95% CI | Gap to warmup, pp |
|---|---:|---:|---:|---:|
| Original CV16, LR 0.0003 | 512 | 84.69 [83.01, 86.34] | 0.98 [-0.20, 2.14] | +7.24 |
| Original CV16, LR 0.001 | 256 | 82.43 [80.62, 84.28] | -1.29 [-3.03, 0.19] | +4.98 |
| Original CV16, LR 0.001 | 512 | 83.85 [82.07, 85.48] | 0.13 [-0.88, 1.13] | +6.39 |
| Original CV8, graph | 512 | 83.70 [82.01, 85.39] | -0.02 [-1.68, 1.50] | +6.25 |
| State256 CV16 first 32, then Oracle16 | 256 | 84.14 [82.54, 85.72] | 0.42 [-0.44, 1.27] | +6.69 |
| Oracle32, graph, LR 0.0003 | 256 | 84.35 [82.64, 86.01] | 0.63 [-0.15, 1.42] | +6.90 |
| Oracle32, LR 0.001 | 128 | 83.81 [82.28, 85.35] | 0.10 [-1.26, 1.43] | +6.36 |
| Oracle32, LR 0.001 | 256 | 83.72 [82.06, 85.46] | 0.00 [0.00, 0.00] | +6.26 |
| Oracle32, LR 0.001 | 512 | 83.63 [82.04, 85.21] | -0.08 [-1.31, 1.10] | +6.18 |
| Oracle16, LR 0.0003 | 512 | 84.50 [82.84, 86.16] | 0.78 [-0.24, 1.82] | +7.04 |
| Oracle8, graph | 512 | 83.52 [81.71, 85.33] | -0.20 [-2.12, 1.72] | +6.07 |
| Oracle16, LR 0.001 | 256 | 83.78 [82.24, 85.32] | 0.07 [-0.84, 0.93] | +6.33 |
| Oracle16, LR 0.001 | 512 | 83.83 [82.33, 85.37] | 0.11 [-1.39, 1.46] | +6.38 |
| Zero / Adam, true every 4, LR 0.001 | 256 | 83.37 [81.79, 85.04] | -0.34 [-1.89, 0.94] | +5.92 |

Warmup-only mean generated accuracy: **77.45%**. The fullpool reference is Oracle32/256/LR 0.001; candidate references are Oracle32/128/LR 0.0003. Candidate quality averages the five fixed subsets within each start; they are not independent seed replicates.

| Method | Updates | Qualified prep + updates, s | Qualified fresh process, s | Historical prep + updates, s | Recorded offline components, s | Profiled matrix PFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Original CV16, LR 0.0003 | 512 | 37.44 | 46.19 | 38.00 | 124.00 | 5.6008 |
| Original CV16, LR 0.001 | 256 | 20.76 | 29.27 | 21.47 | 124.00 | 3.0381 |
| Original CV16, LR 0.001 | 512 | 37.31 | 46.02 | 38.35 | 124.00 | 5.6008 |
| Original CV8, graph | 512 | 24.54 | 33.54 | 25.25 | 124.00 | 3.5234 |
| State256 CV16 first 32, then Oracle16 | 256 | 19.46 | 28.03 | 19.85 | 488.83 | 3.0162 |
| Oracle32, graph, LR 0.0003 | 256 | 30.06 | 39.00 | 30.38 | 0.00 | 6.8481 |
| Oracle32, LR 0.001 | 128 | 15.61 | 24.08 | 16.27 | 0.00 | 3.1931 |
| Oracle32, LR 0.001 | 256 | 27.38 | 36.09 | 28.10 | 0.00 | 5.8821 |
| Oracle32, LR 0.001 | 512 | 50.86 | 59.69 | 52.01 | 0.00 | 11.2631 |
| Oracle16, LR 0.0003 | 512 | 34.13 | 42.89 | 34.44 | 0.00 | 5.5533 |
| Oracle8, graph | 512 | 21.11 | 30.03 | 21.68 | 0.00 | 3.4759 |
| Oracle16, LR 0.001 | 256 | 18.86 | 27.51 | 19.62 | 0.00 | 3.0144 |
| Oracle16, LR 0.001 | 512 | 33.68 | 42.33 | 35.03 | 0.00 | 5.5533 |
| Zero / Adam, true every 4, LR 0.001 | 256 | 10.17 | 18.62 | 10.34 | 0.00 | 1.8511 |

Qualified times average per-start medians of three fresh-process, exact-weight replays. Fullpool timing covers all three starts; candidate timing is representative **candidate0 only**, whereas historical costs cover all five subsets. Dashes indicate no qualified measurement. Fresh-process times include loading/imports/final save; both timing columns exclude the shared target warmup, offline construction and evaluation. Offline totals include recorded source warmups and dependencies, with historical timer boundaries. FLOPs are separately profiled on start 302 and count matrix operations, not all arithmetic.

## Five 1,024-example random training subsets

| Method | Updates | Holdout accuracy %, 95% CI | Gap to population Oracle, pp, 95% CI | Gap to warmup, pp |
|---|---:|---:|---:|---:|
| Original CV16, LR 0.001 | 128 | 82.34 [80.89, 83.86] | -0.77 [-1.26, -0.28] | +4.89 |
| Original CV16, LR 0.001 | 256 | 82.51 [81.02, 83.96] | -0.61 [-1.58, 0.42] | +5.05 |
| Original CV16, LR 0.001 | 512 | 82.14 [80.81, 83.55] | -0.97 [-1.74, -0.21] | +4.69 |
| Oracle32, LR 0.0003 | 128 | 83.11 [81.64, 84.66] | 0.00 [0.00, 0.00] | +5.66 |
| Oracle32, LR 0.001 | 128 | 82.38 [81.03, 83.78] | -0.73 [-1.40, -0.06] | +4.93 |
| Oracle32, LR 0.001 | 256 | 81.82 [80.42, 83.32] | -1.29 [-2.29, -0.40] | +4.36 |
| Oracle32, LR 0.001 | 512 | 82.16 [80.67, 83.68] | -0.95 [-1.66, -0.27] | +4.71 |
| Oracle32, LR 0.001 | 64 | 83.02 [81.58, 84.50] | -0.09 [-0.78, 0.62] | +5.57 |
| Oracle16, LR 0.0003 | 256 | 82.89 [81.49, 84.35] | -0.22 [-0.65, 0.23] | +5.44 |
| Oracle16, LR 0.001 | 128 | 82.21 [80.76, 83.68] | -0.90 [-1.49, -0.31] | +4.76 |
| Oracle16, LR 0.001 | 256 | 82.64 [81.30, 84.04] | -0.47 [-1.16, 0.21] | +5.19 |
| Oracle16, LR 0.001 | 512 | 82.40 [81.05, 83.78] | -0.71 [-1.41, 0.02] | +4.95 |
| Zero / Adam, true every 4, LR 0.0003 | 256 | 82.61 [81.16, 84.07] | -0.50 [-0.92, -0.08] | +5.15 |

Warmup-only mean generated accuracy: **77.45%**. The fullpool reference is Oracle32/256/LR 0.001; candidate references are Oracle32/128/LR 0.0003. Candidate quality averages the five fixed subsets within each start; they are not independent seed replicates.

| Method | Updates | Qualified prep + updates, s | Qualified fresh process, s | Historical prep + updates, s | Recorded offline components, s | Profiled matrix PFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Original CV16, LR 0.001 | 128 | 10.20 | 17.86 | 10.15 | 124.00 | 1.3964 |
| Original CV16, LR 0.001 | 256 | — | — | 18.39 | 124.00 | — |
| Original CV16, LR 0.001 | 512 | — | — | 34.91 | 124.00 | — |
| Oracle32, LR 0.0003 | 128 | 13.62 | 21.34 | 13.80 | 0.00 | 2.8374 |
| Oracle32, LR 0.001 | 128 | — | — | 13.79 | 0.00 | — |
| Oracle32, LR 0.001 | 256 | — | — | 25.61 | 0.00 | — |
| Oracle32, LR 0.001 | 512 | — | — | 49.25 | 0.00 | — |
| Oracle32, LR 0.001 | 64 | 7.76 | 15.52 | 7.87 | 0.00 | 1.4816 |
| Oracle16, LR 0.0003 | 256 | 16.84 | 24.75 | 16.81 | 0.00 | 2.6223 |
| Oracle16, LR 0.001 | 128 | 9.33 | 17.01 | 9.16 | 0.00 | 1.3846 |
| Oracle16, LR 0.001 | 256 | — | — | 16.67 | 0.00 | — |
| Oracle16, LR 0.001 | 512 | — | — | 31.65 | 0.00 | — |
| Zero / Adam, true every 4, LR 0.0003 | 256 | 8.08 | 16.01 | 8.24 | 0.00 | 1.4756 |

Qualified times average per-start medians of three fresh-process, exact-weight replays. Fullpool timing covers all three starts; candidate timing is representative **candidate0 only**, whereas historical costs cover all five subsets. Dashes indicate no qualified measurement. Fresh-process times include loading/imports/final save; both timing columns exclude the shared target warmup, offline construction and evaluation. Offline totals include recorded source warmups and dependencies, with historical timer boundaries. FLOPs are separately profiled on start 302 and count matrix operations, not all arithmetic.

## Five 1,024-example source-disjoint genre training subsets

| Method | Updates | Holdout accuracy %, 95% CI | Gap to population Oracle, pp, 95% CI | Gap to warmup, pp |
|---|---:|---:|---:|---:|
| Original CV16, LR 0.001 | 128 | 82.50 [80.93, 84.03] | -0.86 [-1.43, -0.36] | +5.05 |
| Original CV16, LR 0.001 | 256 | 82.17 [80.77, 83.56] | -1.19 [-1.75, -0.59] | +4.71 |
| Original CV16, LR 0.001 | 512 | 81.62 [80.25, 83.02] | -1.74 [-2.46, -0.98] | +4.17 |
| Oracle32, LR 0.0003 | 128 | 83.36 [81.82, 84.91] | 0.00 [0.00, 0.00] | +5.91 |
| Oracle32, LR 0.001 | 128 | 82.33 [80.92, 83.73] | -1.03 [-1.58, -0.49] | +4.88 |
| Oracle32, LR 0.001 | 256 | 81.93 [80.51, 83.35] | -1.43 [-2.04, -0.82] | +4.48 |
| Oracle32, LR 0.001 | 512 | 81.43 [79.74, 83.05] | -1.93 [-3.10, -0.83] | +3.98 |
| Oracle32, LR 0.001 | 64 | 82.90 [81.40, 84.39] | -0.46 [-1.06, 0.15] | +5.45 |
| Oracle16, LR 0.0003 | 256 | 82.97 [81.52, 84.42] | -0.39 [-0.89, 0.13] | +5.52 |
| Oracle16, LR 0.001 | 128 | 82.52 [81.02, 84.01] | -0.85 [-1.39, -0.33] | +5.06 |
| Oracle16, LR 0.001 | 256 | 82.39 [80.95, 83.82] | -0.97 [-1.57, -0.36] | +4.94 |
| Oracle16, LR 0.001 | 512 | 81.65 [80.27, 83.02] | -1.71 [-2.38, -0.98] | +4.20 |
| Zero / Adam, true every 4, LR 0.0003 | 256 | 82.94 [81.39, 84.47] | -0.42 [-0.76, -0.09] | +5.49 |

Warmup-only mean generated accuracy: **77.45%**. The fullpool reference is Oracle32/256/LR 0.001; candidate references are Oracle32/128/LR 0.0003. Candidate quality averages the five fixed subsets within each start; they are not independent seed replicates.

| Method | Updates | Qualified prep + updates, s | Qualified fresh process, s | Historical prep + updates, s | Recorded offline components, s | Profiled matrix PFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Original CV16, LR 0.001 | 128 | — | — | 10.00 | 124.00 | — |
| Original CV16, LR 0.001 | 256 | — | — | 18.21 | 124.00 | — |
| Original CV16, LR 0.001 | 512 | — | — | 34.73 | 124.00 | — |
| Oracle32, LR 0.0003 | 128 | — | — | 13.54 | 0.00 | — |
| Oracle32, LR 0.001 | 128 | — | — | 13.53 | 0.00 | — |
| Oracle32, LR 0.001 | 256 | — | — | 25.13 | 0.00 | — |
| Oracle32, LR 0.001 | 512 | — | — | 48.31 | 0.00 | — |
| Oracle32, LR 0.001 | 64 | — | — | 7.75 | 0.00 | — |
| Oracle16, LR 0.0003 | 256 | — | — | 16.94 | 0.00 | — |
| Oracle16, LR 0.001 | 128 | — | — | 9.16 | 0.00 | — |
| Oracle16, LR 0.001 | 256 | — | — | 16.69 | 0.00 | — |
| Oracle16, LR 0.001 | 512 | — | — | 31.72 | 0.00 | — |
| Zero / Adam, true every 4, LR 0.0003 | 256 | — | — | 8.12 | 0.00 | — |

Qualified times average per-start medians of three fresh-process, exact-weight replays. Fullpool timing covers all three starts; candidate timing is representative **candidate0 only**, whereas historical costs cover all five subsets. Dashes indicate no qualified measurement. Fresh-process times include loading/imports/final save; both timing columns exclude the shared target warmup, offline construction and evaluation. Offline totals include recorded source warmups and dependencies, with historical timer boundaries. FLOPs are separately profiled on start 302 and count matrix operations, not all arithmetic.

## Additional paired controls

| Population | Actual recipe | Reference recipe | Accuracy difference, pp, 95% CI |
|---|---|---|---:|
| fullpool | Original CV8, graph / 512 | Oracle8, graph / 512 | 0.18 [-1.01, 1.30] |
| fullpool | Original CV8, graph / 512 | Original CV16, LR 0.0003 / 512 | -0.99 [-2.66, 0.59] |
| fullpool | Original CV8, graph / 512 | Zero / Adam, true every 4, LR 0.001 / 256 | 0.33 [-0.73, 1.37] |
| fullpool | Original CV8, graph / 512 | Oracle32, graph, LR 0.0003 / 256 | -0.65 [-2.26, 0.78] |
| fullpool | State256 CV16 first 32, then Oracle16 / 256 | Oracle16, LR 0.001 / 256 | 0.36 [-0.30, 0.95] |
| fullpool | State256 CV16 first 32, then Oracle16 / 256 | Original CV16, LR 0.001 / 256 | 1.71 [0.26, 3.34] |
| fullpool | State256 CV16 first 32, then Oracle16 / 256 | Zero / Adam, true every 4, LR 0.001 / 256 | 0.76 [-0.48, 2.14] |
| fullpool | State256 CV16 first 32, then Oracle16 / 256 | Oracle32, graph, LR 0.0003 / 256 | -0.21 [-1.04, 0.59] |
| fullpool | Original CV16, LR 0.0003 / 512 | Oracle32, graph, LR 0.0003 / 256 | 0.34 [-0.74, 1.45] |
| random | Original CV16, LR 0.001 / 128 | Oracle16, LR 0.001 / 128 | 0.13 [-0.29, 0.53] |
| random | Original CV16, LR 0.001 / 128 | Oracle32, LR 0.001 / 64 | -0.68 [-1.41, 0.04] |
| random | Original CV16, LR 0.001 / 128 | Zero / Adam, true every 4, LR 0.0003 / 256 | -0.26 [-0.78, 0.25] |
| genre | Original CV16, LR 0.001 / 128 | Oracle16, LR 0.001 / 128 | -0.02 [-0.52, 0.50] |
| genre | Original CV16, LR 0.001 / 128 | Oracle32, LR 0.001 / 64 | -0.40 [-1.25, 0.35] |
| genre | Original CV16, LR 0.001 / 128 | Zero / Adam, true every 4, LR 0.0003 / 256 | -0.44 [-1.03, 0.11] |

These paired comparisons separate predictor contribution, shorter/cheaper true-gradient alternatives, and the additional graph Oracle that had higher development accuracy. The primary population-specific Oracle references remain unchanged.

## Candidate-selection outcomes

Each row selects one of five datasets using selection-dev, then reports the chosen **reference Oracle** agent on selection-holdout. This is distinct from deploying the predictor-updated agent directly. Cost estimates include the final reference-Oracle training run unless that exact adapter was already trained while screening.

### Random candidates: aggregate target

| Simulator / readout | Updates | Chosen Oracle accuracy % | Gain vs random, pp, 95% CI | Gap vs Oracle-dev choice, pp, 95% CI | Regret to holdout ceiling, pp | Ready component sum, s |
|---|---:|---:|---:|---:|---:|---:|
| Original CV16, LR 0.001 / exact_teacher_forced_accuracy | 128 | 83.60 | 0.49 [-0.14, 1.09] | 0.16 [-0.86, 1.21] | 0.54 | 72.93 |
| Original CV16, LR 0.001 / exact_teacher_gold_probability | 128 | 83.28 | 0.17 [-0.43, 0.81] | -0.16 [-1.00, 0.64] | 0.86 | 72.99 |
| Original CV16, LR 0.001 / generated_accuracy | 128 | 83.60 | 0.49 [-0.14, 1.09] | 0.16 [-0.86, 1.21] | 0.54 | 81.71 |
| Original CV16, LR 0.001 / Anchored128 / forced_accuracy | 128 | 83.16 | 0.05 [-0.50, 0.58] | -0.28 [-1.16, 0.57] | 0.98 | 68.28 |
| Original CV16, LR 0.001 / Anchored128 / gold_probability | 128 | 83.20 | 0.08 [-0.49, 0.64] | -0.24 [-1.08, 0.58] | 0.94 | 68.28 |
| Oracle32, LR 0.0003 / exact_teacher_forced_accuracy | 128 | 83.44 | 0.33 [-0.42, 1.01] | 0.00 [0.00, 0.00] | 0.70 | 77.42 |
| Oracle32, LR 0.0003 / exact_teacher_gold_probability | 128 | 83.65 | 0.54 [-0.35, 1.47] | 0.21 [-0.82, 1.46] | 0.49 | 77.42 |
| Oracle32, LR 0.0003 / generated_accuracy | 128 | 83.44 | 0.33 [-0.42, 1.01] | 0.00 [0.00, 0.00] | 0.70 | 86.16 |
| Oracle32, LR 0.0003 / Anchored128 / forced_accuracy | 128 | 83.47 | 0.36 [-0.30, 1.00] | 0.03 [-1.06, 1.21] | 0.67 | 72.69 |
| Oracle32, LR 0.0003 / Anchored128 / gold_probability | 128 | 83.54 | 0.43 [-0.34, 1.10] | 0.10 [-0.33, 0.69] | 0.60 | 72.69 |
| Oracle32, LR 0.001 / exact_teacher_forced_accuracy | 64 | 83.20 | 0.08 [-0.65, 0.80] | -0.24 [-1.21, 0.73] | 0.94 | 61.57 |
| Oracle32, LR 0.001 / exact_teacher_gold_probability | 64 | 83.29 | 0.18 [-0.36, 0.78] | -0.15 [-1.09, 0.80] | 0.85 | 61.72 |
| Oracle32, LR 0.001 / generated_accuracy | 64 | 83.20 | 0.08 [-0.65, 0.80] | -0.24 [-1.21, 0.73] | 0.94 | 70.31 |
| Oracle32, LR 0.001 / Anchored128 / forced_accuracy | 64 | 83.16 | 0.05 [-0.50, 0.58] | -0.28 [-1.16, 0.57] | 0.98 | 56.88 |
| Oracle32, LR 0.001 / Anchored128 / gold_probability | 64 | 83.20 | 0.08 [-0.49, 0.64] | -0.24 [-1.08, 0.58] | 0.94 | 56.88 |
| Zero / Adam, true every 4, LR 0.0003 / exact_teacher_forced_accuracy | 256 | 81.99 | -1.12 [-2.32, -0.01] | -1.45 [-2.99, 0.11] | 2.15 | 63.45 |
| Zero / Adam, true every 4, LR 0.0003 / exact_teacher_gold_probability | 256 | 84.14 | 1.03 [0.46, 1.66] | 0.70 [0.00, 1.69] | 0.00 | 63.33 |
| Zero / Adam, true every 4, LR 0.0003 / generated_accuracy | 256 | 82.45 | -0.66 [-2.05, 0.39] | -0.99 [-2.59, 0.33] | 1.69 | 72.39 |
| Zero / Adam, true every 4, LR 0.0003 / Anchored128 / forced_accuracy | 256 | 83.20 | 0.08 [-0.50, 0.65] | -0.24 [-1.17, 0.82] | 0.94 | 58.90 |
| Zero / Adam, true every 4, LR 0.0003 / Anchored128 / gold_probability | 256 | 84.14 | 1.03 [0.46, 1.66] | 0.70 [0.00, 1.69] | 0.00 | 58.60 |

Training the original full 4,097-example pool once reaches **83.72%** without candidate screening. Qualified preparation/update time: **27.38s**. This permits more examples than the 1,024-example candidate constraint; it is a practical alternative, not a same-data-size ablation.

Ready selection costs are **recorded component sums, not measured end-to-end pipeline latency**. Shared engine/model startup, process transitions and unrecorded I/O remain additional; fitted-predictor costs are separate in the cost ledger. Forced ABC scores and free-generation accuracy use different readouts; exact-teacher forced-score controls are included. The best-holdout candidate is a nondeployable diagnostic ceiling.

### Genre candidates: aggregate target

| Simulator / readout | Updates | Chosen Oracle accuracy % | Gain vs random, pp, 95% CI | Gap vs Oracle-dev choice, pp, 95% CI | Regret to holdout ceiling, pp | Ready component sum, s |
|---|---:|---:|---:|---:|---:|---:|
| Original CV16, LR 0.001 / exact_teacher_forced_accuracy | 128 | 83.93 | 0.57 [-0.33, 1.43] | 0.00 [-1.23, 1.22] | 0.42 | 72.50 |
| Original CV16, LR 0.001 / exact_teacher_gold_probability | 128 | 83.23 | -0.13 [-1.23, 0.88] | -0.70 [-2.07, 0.36] | 1.12 | 72.70 |
| Original CV16, LR 0.001 / generated_accuracy | 128 | 83.93 | 0.57 [-0.33, 1.43] | 0.00 [-1.23, 1.22] | 0.42 | 81.17 |
| Original CV16, LR 0.001 / Anchored128 / forced_accuracy | 128 | 83.28 | -0.08 [-1.06, 0.85] | -0.65 [-1.71, 0.39] | 1.07 | 66.12 |
| Original CV16, LR 0.001 / Anchored128 / gold_probability | 128 | 82.90 | -0.46 [-1.33, 0.29] | -1.02 [-2.33, 0.18] | 1.45 | 66.03 |
| Oracle32, LR 0.0003 / exact_teacher_forced_accuracy | 128 | 83.99 | 0.63 [-0.07, 1.29] | 0.07 [-0.57, 0.78] | 0.36 | 76.12 |
| Oracle32, LR 0.0003 / exact_teacher_gold_probability | 128 | 83.94 | 0.58 [-0.11, 1.33] | 0.02 [-0.71, 0.74] | 0.41 | 76.12 |
| Oracle32, LR 0.0003 / generated_accuracy | 128 | 83.93 | 0.57 [-0.11, 1.20] | 0.00 [0.00, 0.00] | 0.42 | 84.71 |
| Oracle32, LR 0.0003 / Anchored128 / forced_accuracy | 128 | 83.54 | 0.18 [-0.59, 0.90] | -0.39 [-1.46, 0.44] | 0.81 | 71.39 |
| Oracle32, LR 0.0003 / Anchored128 / gold_probability | 128 | 83.00 | -0.36 [-0.95, 0.25] | -0.93 [-1.79, -0.07] | 1.35 | 71.39 |
| Oracle32, LR 0.001 / exact_teacher_forced_accuracy | 64 | 83.88 | 0.52 [-0.16, 1.19] | -0.05 [-0.90, 0.72] | 0.47 | 60.32 |
| Oracle32, LR 0.001 / exact_teacher_gold_probability | 64 | 83.75 | 0.39 [-0.61, 1.33] | -0.18 [-1.05, 0.62] | 0.60 | 62.24 |
| Oracle32, LR 0.001 / generated_accuracy | 64 | 83.88 | 0.52 [-0.16, 1.19] | -0.05 [-0.90, 0.72] | 0.47 | 69.01 |
| Oracle32, LR 0.001 / Anchored128 / forced_accuracy | 64 | 83.52 | 0.16 [-0.61, 0.97] | -0.41 [-1.49, 0.68] | 0.83 | 55.72 |
| Oracle32, LR 0.001 / Anchored128 / gold_probability | 64 | 83.93 | 0.57 [-0.37, 1.42] | 0.00 [-1.35, 1.30] | 0.42 | 55.60 |
| Zero / Adam, true every 4, LR 0.0003 / exact_teacher_forced_accuracy | 256 | 83.57 | 0.21 [-0.52, 0.95] | -0.36 [-1.38, 0.45] | 0.78 | 63.37 |
| Zero / Adam, true every 4, LR 0.0003 / exact_teacher_gold_probability | 256 | 83.33 | -0.04 [-0.84, 0.85] | -0.60 [-1.55, 0.10] | 1.02 | 63.20 |
| Zero / Adam, true every 4, LR 0.0003 / generated_accuracy | 256 | 83.52 | 0.16 [-0.61, 0.97] | -0.41 [-1.49, 0.68] | 0.83 | 70.90 |
| Zero / Adam, true every 4, LR 0.0003 / Anchored128 / forced_accuracy | 256 | 83.63 | 0.27 [-1.14, 1.42] | -0.29 [-2.01, 1.30] | 0.72 | 57.54 |
| Zero / Adam, true every 4, LR 0.0003 / Anchored128 / gold_probability | 256 | 83.63 | 0.27 [-1.11, 1.43] | -0.29 [-1.97, 1.21] | 0.72 | 58.45 |

Ready selection costs are **recorded component sums, not measured end-to-end pipeline latency**. Shared engine/model startup, process transitions and unrecorded I/O remain additional; fitted-predictor costs are separate in the cost ledger. Forced ABC scores and free-generation accuracy use different readouts; exact-teacher forced-score controls are included. The best-holdout candidate is a nondeployable diagnostic ceiling.

## Keeping the screened adapter directly

This uses the already-frozen secondary `direct_deploy_accuracy` endpoint: choose on selection-dev and deploy that simulator-produced adapter, without a final reference-Oracle retraining. All rows below use generated-accuracy selection. The chosen dataset and its adapter are fixed before holdout readouts. This answers a different deployment question from the chosen-reference-Oracle table above.

| Candidates | Screening method | Updates | Deployed accuracy % | Gap vs Oracle-dev selected agent, pp, 95% CI | Ready component sum without final retraining, s | Offline components, s |
|---|---|---:|---:|---:|---:|---:|
| genre | Original CV16, LR 0.001 | 128 | 83.78 | -0.15 [-1.27, 0.97] | 67.07 | 124.00 |
| genre | Oracle32, LR 0.0003 | 128 | 83.93 | 0.00 [0.00, 0.00] | 84.71 | 0.00 |
| genre | Oracle32, LR 0.001 | 64 | 83.47 | -0.46 [-1.60, 0.55] | 55.81 | 0.00 |
| genre | Oracle16, LR 0.001 | 128 | 82.58 | -1.35 [-2.56, -0.10] | 62.73 | 0.00 |
| genre | Zero / Adam, true every 4, LR 0.0003 | 256 | 82.53 | -1.40 [-2.92, 0.11] | 57.59 | 0.00 |
| random | Original CV16, LR 0.001 | 128 | 83.21 | -0.23 [-1.25, 0.85] | 67.92 | 124.00 |
| random | Oracle32, LR 0.0003 | 128 | 83.44 | 0.00 [0.00, 0.00] | 86.16 | 0.00 |
| random | Oracle32, LR 0.001 | 64 | 84.32 | 0.88 [-0.59, 2.22] | 56.50 | 0.00 |
| random | Oracle16, LR 0.001 | 128 | 82.46 | -0.98 [-2.88, 0.98] | 62.97 | 0.00 |
| random | Zero / Adam, true every 4, LR 0.0003 | 256 | 82.51 | -0.93 [-1.98, 0.02] | 58.47 | 0.00 |

These costs subtract the separately recorded final-Oracle component from the existing component ledger; they are not separately measured complete-workflow latencies. The measured workflow benchmark below retains its registered final-reference-Oracle endpoint. Reporting these frozen secondary results does not select a new winning method from holdout outcomes.

## Measured complete selection workflows

One representative start 302 and the five random candidates; three independent timing repetitions per condition. Every candidate adaptation is rerun, every readout is freshly computed, and the final chosen Oracle is rerun when necessary. Final weights, readouts and choices must exactly reproduce the frozen references.

| Workflow | Median total seconds | Range, seconds | Repeats |
|---|---:|---:|---:|
| Oracle32 + generation | 152.15 | 151.85–152.57 | 3 |
| Oracle32 + exact ABC score | 132.05 | 131.51–133.33 | 3 |
| CV16 + generation | 157.93 | 155.31–159.98 | 3 |
| CV16 + exact ABC score | 136.56 | 136.44–140.84 | 3 |
| CV16 + learned behavior score | 133.65 | 132.50–133.66 | 3 |
| Short Oracle32 + exact ABC score | 123.93 | 123.80–124.78 | 3 |

These are actual fresh-process workflow times, including loading, process transitions, serialization and equality checks. Offline predictor construction and the common target warmup remain additional. The implementation launches separate adaptation/readout processes; it is not an optimized persistent model server. Three timing repetitions are not three scientific starts. Genre workflows are not timed end-to-end.

## Predicting changed agent behavior

| Fullpool agent | Updates | Exact teacher % | Behavior surrogate % | Unchanged initial % | Agreement gain vs unchanged, pp, 95% CI |
|---|---:|---:|---:|---:|---:|
| Original CV16, LR 0.001 | 256 | 82.40 | 76.02 | 77.62 | -1.42 [-3.25, 0.34] |
| Original CV8, graph | 512 | 83.68 | 76.79 | 77.62 | -1.04 [-2.70, 0.51] |
| State256 CV16 first 32, then Oracle16 | 256 | 84.20 | 76.74 | 77.62 | -1.68 [-2.90, -0.65] |
| Oracle32, LR 0.001 | 256 | 83.72 | 76.96 | 77.62 | -1.07 [-3.39, 0.85] |

Forced-format ABC outcomes only. Positive agreement gain means better prediction of the current teacher than simply retaining the initial teacher. Full logit-error and class-change diagnostics are in [behavior fidelity](results/final_behavior_fidelity.json).

All method/horizon/readout/genre-target results, candidate spreads and per-start choices are in [selection analysis](results/final_selection_analysis.json), [cost ledger](results/surrogate_cost_comparison.json), and [qualified timing receipts](results/standalone_reuse_timing_summary.json). Cheap constant, random and matching-genre metadata selectors are included in those files.
