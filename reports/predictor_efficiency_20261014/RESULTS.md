# Frozen results

Recipes and stopping points were selected on development data and frozen before these readouts. Each trained cohort uses update seeds 101/102/103; warmup is one shared starting adapter. These evaluation populations were used in earlier studies, so the evidence is exploratory.

Accuracy intervals and paired gaps resample update seeds and whole source groups. The practical margin is −1 percentage point; a point estimate within that margin alone does not establish noninferiority.

## PAWS

| Method | Steps | LR | Test accuracy % [95% CI] | Gap to Oracle, pp [95% CI] |
|---|---:|---:|---:|---:|
| Oracle32, cached prefix | 512 | 0.0003 | 89.79 [88.59, 90.95] | Reference |
| Oracle16 | 256 | 0.0003 | 88.68 [87.59, 89.78] | -1.10 [-2.18, 0.21] |
| Oracle32, early stop | 128 | 0.001 | 88.87 [87.88, 89.86] | -0.92 [-1.70, -0.03] |
| CV16 (primary predictor) | 256 | 0.0003 | 88.61 [87.54, 89.70] | -1.18 [-2.21, 0.06] |
| CV4 | 256 | 0.0003 | 87.22 [86.09, 88.33] | -2.57 [-3.84, -1.28] |
| CV8, long run | 1024 | 0.0003 | 90.02 [89.05, 90.99] | 0.23 [-0.63, 1.30] |
| Gradient hold, period 2 | 256 | 0.0003 | 88.67 [87.57, 89.80] | -1.12 [-2.31, 0.30] |
| Predictor anchor, period 2 | 256 | 0.0003 | 88.57 [87.44, 89.76] | -1.22 [-2.49, 0.33] |
| Warmup | 0 | 0.001 | 84.39 [83.15, 85.54] | -5.40 [-6.58, -4.14] |

| Method | Online + preparation, s | Standalone script, s [range] | Recorded predictor preparation, s | Matrix PFLOPs incl. cache | Replay weights match? |
|---|---:|---:|---:|---:|---|
| Oracle32, cached prefix | 46.30 | 55.23 [54.52, 55.91] | 0.00 | 9.861 | Yes, all 3 registered seeds |
| Oracle16 | 17.98 | 27.02 [26.39, 27.27] | 0.00 | — | Yes, all 3 registered seeds |
| Oracle32, early stop | 13.54 | 22.30 [22.23, 22.47] | 0.00 | — | Yes, all 3 registered seeds |
| CV16 | 20.35 | 29.22 [28.86, 29.23] | 85.11 | 2.704 | Yes, all 3 registered seeds |
| CV4 | 20.69 | 29.53 [29.14, 29.60] | 85.11 | 1.059 | Yes, all 3 registered seeds |
| CV8, long run | 72.85 | 82.32 [81.26, 83.28] | 85.11 | — | Yes, all 3 registered seeds |
| Gradient hold, period 2 | 14.02 | 22.47 [22.34, 23.02] | 0.00 | — | Yes, all 3 registered seeds |
| Predictor anchor, period 2 | 18.14 | 26.81 [26.72, 27.20] | 85.11 | — | Yes, all 3 registered seeds |

## MNLI

| Method | Steps | LR | Test accuracy % [95% CI] | Gap to Oracle, pp [95% CI] |
|---|---:|---:|---:|---:|
| Oracle32 | 512 | 0.0003 | 84.82 [84.04, 85.57] | Reference |
| Oracle32, cached prefix | 512 | 0.0003 | 84.92 [84.18, 85.65] | 0.11 [-0.03, 0.24] |
| Oracle32, early stop | 256 | 0.001 | 84.97 [84.04, 85.79] | 0.15 [-0.33, 0.59] |
| Oracle8 | 512 | 0.001 | 84.04 [83.33, 84.77] | -0.77 [-1.27, -0.31] |
| Online class RLS8 (primary predictor) | 512 | 0.001 | 84.49 [83.51, 85.42] | -0.33 [-0.87, 0.23] |
| CV8 | 512 | 0.001 | 83.33 [82.39, 84.23] | -1.48 [-2.45, -0.68] |
| CV16 | 256 | 0.0003 | 84.68 [83.93, 85.43] | -0.13 [-0.65, 0.37] |
| CV4 | 256 | 0.0003 | 82.76 [81.40, 83.92] | -2.05 [-3.53, -0.79] |
| Warmup | 0 | 0.001 | 78.24 [77.42, 79.03] | -6.58 [-7.38, -5.78] |

| Method | Online + preparation, s | Standalone script, s [range] | Recorded predictor preparation, s | Matrix PFLOPs incl. cache | Replay weights match? |
|---|---:|---:|---:|---:|---|
| Oracle32 | 56.57 | 65.66 [65.60, 66.38] | 0.00 | 12.975 | Yes, all 3 registered seeds |
| Oracle32, cached prefix | 51.06 | 60.33 [60.20, 60.42] | 0.00 | 11.311 | Yes, all 3 registered seeds |
| Oracle32, early stop | 28.93 | 37.95 [37.45, 38.00] | 0.00 | 6.497 | Yes, all 3 registered seeds |
| Oracle8 | 32.25 | 41.18 [40.53, 41.29] | 0.00 | — | Yes, all 3 registered seeds |
| Online class RLS8 | 35.95 | 44.98 [44.93, 45.09] | 0.00 | 2.816 | Yes, all 3 registered seeds |
| CV8 | 39.77 | 48.94 [48.01, 49.59] | 110.26 | 3.363 | Yes, all 3 registered seeds |
| CV16 | 23.20 | 32.17 [32.16, 32.61] | 110.26 | 3.552 | Yes, all 3 registered seeds |
| CV4 | 20.98 | 29.73 [29.71, 30.39] | 110.26 | 1.176 | Yes, all 3 registered seeds |

## Confirmation readout

All frozen cohorts are also evaluated on the 512-example confirmation split for each task. These smaller readouts have wider uncertainty; no recipe was changed after seeing them.

| Task | Method | Accuracy % | Gap to selected Oracle32, pp [95% CI] |
|---|---|---:|---:|
| MNLI | Oracle32 | 83.20 | Reference |
| MNLI | Oracle32, cached prefix | 83.27 | 0.07 [-0.65, 0.85] |
| MNLI | Oracle32, early stop | 83.07 | -0.13 [-2.35, 1.95] |
| MNLI | Oracle8 | 82.16 | -1.04 [-3.51, 1.17] |
| MNLI | Online class RLS8 | 82.36 | -0.85 [-2.74, 1.04] |
| MNLI | CV8 | 81.58 | -1.63 [-3.78, 0.46] |
| MNLI | CV16 | 82.75 | -0.46 [-2.94, 1.56] |
| MNLI | CV4 | 80.40 | -2.80 [-6.32, 0.20] |
| MNLI | Warmup | 74.61 | -8.59 [-11.76, -5.46] |
| PAWS | Oracle32, cached prefix | 89.32 | Reference |
| PAWS | Oracle16 | 88.41 | -0.91 [-3.80, 1.84] |
| PAWS | Oracle32, early stop | 88.28 | -1.04 [-3.46, 1.26] |
| PAWS | CV16 | 88.48 | -0.85 [-3.91, 2.07] |
| PAWS | CV4 | 85.09 | -4.23 [-8.27, -0.06] |
| PAWS | CV8, long run | 89.52 | 0.20 [-2.84, 3.40] |
| PAWS | Gradient hold, period 2 | 88.09 | -1.24 [-4.47, 2.20] |
| PAWS | Predictor anchor, period 2 | 88.28 | -1.04 [-4.15, 2.31] |
| PAWS | Warmup | 81.05 | -8.27 [-13.10, -3.72] |

Times are medians of three clean standalone replays. Online + preparation includes adaptation and required cache/initialization; standalone additionally includes imports, loading, and final serialization. Both exclude inherited predictor preparation, shared warmup training, and evaluation. Recorded predictor preparation is a separate component mean, not a newly timed cold pipeline. A dash means unmeasured.

FLOPs are seed-101 matrix-operation counts for the matching execution module, including cache preparation but excluding inherited predictor fitting and scalar/nonlinear/optimizer arithmetic. The small RLS linear solves are also uncounted; each receipt lists operator coverage. Instrumented runtimes are not used as speed measurements.

Machine-readable values: [CSV](results/final_comparison.csv), [test accuracy and paired intervals](results/frozen_test_analysis.json), [confirmation analysis](results/frozen_oracle_confirmation_analysis.json), [supplementary paired controls](results/frozen_control_comparisons.json), [timing receipts](results/frozen_timing_analysis.json), [preparation ledger](results/predictor_preparation_costs.json).
