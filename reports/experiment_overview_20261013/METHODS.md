# October predictor variants and controls

[Overview](README.md) · [Machine-readable table](recent_method_comparisons.csv)

Every completed final recipe, grouped only when method, LR, stopping and control role match. Single-seed alternatives are not three-seed comparisons. These were selected on development; this table does not promote a method based on final accuracy. Accuracy uses the source-disjoint population (BoolQ 0.5B: 2,713 questions).

## 0.5B

### BoolQ

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 73.09 | +8.77 |
| CV4 | — | 9e-05 | 128 | 3 | 67.82 | +3.50 |
| Adaptive | — | 3e-05 | 256 | 1 | 66.90 | +2.58 |
| Frozen + Batchfit | — | 3e-05 | 8 | 1 | 66.79 | +2.47 |
| Frozen | — | 3e-05 | 8 | 3 | 66.49 | +2.17 |
| Oracle4 (tuned) | — | 9e-05 | 128 | 3 | 67.43 | +3.11 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 69.41 | +5.09 |
| Mean | Earlier predictor recipe | 3e-05 | 8 | 1 | 67.27 | +2.95 |

### MNLI

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 256 | 3 | 80.16 | +8.73 |
| Adaptive | — | 3e-05 | 8 | 3 | 70.84 | -0.58 |
| Frozen | — | 3e-05 | 8 | 1 | 71.36 | -0.06 |
| Mean | Oracle32 recipe | 0.001 | 256 | 1 | 76.11 | +4.69 |
| Mean | Selected predictor recipe | 3e-05 | 8 | 1 | 72.11 | +0.69 |

### PAWS

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 256 | 3 | 85.86 | +13.18 |
| CV4 | — | 0.001 | 128 | 3 | 79.69 | +7.01 |
| Frozen | — | 0.0003 | 8 | 1 | 72.82 | +0.15 |
| Adaptive | — | 0.0001 | 128 | 3 | 71.98 | -0.70 |
| Oracle4 (tuned) | — | 0.001 | 256 | 3 | 82.16 | +9.49 |
| Oracle4 (matched) | — | 0.001 | 128 | 3 | 75.67 | +3.00 |
| Mean | Oracle32 recipe | 0.001 | 256 | 1 | 79.83 | +7.15 |
| Mean | Earlier predictor recipe | 0.0001 | 128 | 1 | 75.89 | +3.21 |

### ARC-Easy

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 8 | 3 | 72.56 | +1.47 |
| Adaptive | — | 3e-05 | 8 | 3 | 71.51 | +0.42 |
| Frozen | — | 3e-05 | 8 | 1 | 71.59 | +0.51 |
| Mean | Oracle32 recipe | 0.001 | 8 | 1 | 71.93 | +0.84 |
| Mean | Selected predictor recipe | 3e-05 | 8 | 1 | 71.13 | +0.04 |

### OpenBookQA

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0001 | 256 | 3 | 60.67 | +3.67 |
| Frozen + Globalfit | — | 3e-05 | 32 | 3 | 58.07 | +1.07 |
| Adaptive | — | 0.0001 | 32 | 1 | 55.80 | -1.20 |
| Frozen | — | 0.0001 | 32 | 3 | 50.07 | -6.93 |
| Mean | Oracle32 recipe | 0.0001 | 256 | 1 | 61.00 | +4.00 |
| Mean | Earlier predictor recipe | 0.0001 | 32 | 1 | 58.00 | +1.00 |
| Mean | Selected predictor recipe | 3e-05 | 32 | 1 | 57.60 | +0.60 |

### HellaSwag

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 256 | 3 | 56.06 | +14.14 |
| Frozen | — | 0.0003 | 8 | 3 | 43.36 | +1.45 |
| Adaptive | — | 0.0003 | 8 | 1 | 43.27 | +1.35 |
| Mean | Oracle32 recipe | 0.001 | 256 | 1 | 47.54 | +5.63 |
| Mean | Selected predictor recipe | 0.0003 | 8 | 1 | 42.38 | +0.47 |

### RACE

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 128 | 3 | 63.45 | +2.61 |
| Adaptive | — | 0.0001 | 8 | 3 | 61.26 | +0.41 |
| Frozen | — | 3e-05 | 8 | 1 | 61.07 | +0.22 |
| Mean | Oracle32 recipe | 0.001 | 128 | 1 | 62.51 | +1.66 |
| Mean | Selected predictor recipe | 0.0001 | 8 | 1 | 61.21 | +0.36 |

### ARC-Challenge

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 3e-05 | 8 | 1 | 52.47 | +0.34 |

### SVAMP

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0001 | 8 | 1 | 63.67 | -2.00 |

### AQuA

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 128 | 1 | 22.44 | -9.84 |

### GSM8K

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 8 | 3 | 42.91 | -4.93 |

### Countdown

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 256 | 3 | 12.11 | +3.52 |
| Adaptive | — | 3e-05 | 8 | 3 | 7.03 | -1.56 |
| Frozen | — | 0.0001 | 8 | 1 | 5.27 | -3.32 |
| Mean | Oracle32 recipe | 0.001 | 256 | 1 | 8.01 | -0.59 |
| Mean | Selected predictor recipe | 3e-05 | 8 | 1 | 6.45 | -2.15 |

### MATH-500

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 3e-05 | 32 | 1 | 32.20 | +0.60 |

### ARC-Challenge (expanded pool)

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 51.42 | -0.71 |
| Frozen | — | 9e-06 | 8 | 3 | 52.16 | +0.03 |
| Adaptive | — | 9e-06 | 8 | 1 | 52.22 | +0.09 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 54.52 | +2.39 |
| Mean | Selected predictor recipe | 9e-06 | 8 | 1 | 51.88 | -0.26 |

### SVAMP (lower LR)

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 3e-06 | 8 | 1 | 63.00 | -2.67 |

### AQuA (answer only)

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 1e-05 | 8 | 1 | 33.86 | -3.15 |

### GSM8K (lower LR)

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 1e-05 | 32 | 3 | 48.24 | +0.40 |
| Adaptive | — | 3e-06 | 128 | 3 | 48.27 | +0.43 |
| Frozen | — | 3e-06 | 256 | 1 | 47.84 | +0.00 |
| Mean | Oracle32 recipe | 1e-05 | 32 | 1 | 48.67 | +0.83 |
| Mean | Selected predictor recipe | 3e-06 | 128 | 1 | 47.61 | -0.23 |

### MATH-500 (lower LR)

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 1e-05 | 1 | 1 | 33.00 | +1.40 |

## 1.5B

### MNLI

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 84.81 | +6.57 |
| CV4 | — | 0.0003 | 256 | 3 | 82.71 | +4.47 |
| Frozen | — | 3e-05 | 8 | 1 | 78.29 | +0.05 |
| Adaptive | — | 9e-06 | 8 | 3 | 78.28 | +0.05 |
| Frozen + labels + Globalfit | — | 9e-06 | 32 | 3 | 78.60 | +0.37 |
| Frozen + Globalfit | — | 3e-05 | 8 | 3 | 78.89 | +0.66 |
| Oracle4 (tuned) | — | 0.001 | 256 | 3 | 81.27 | +3.04 |
| Oracle4 (matched) | — | 0.0003 | 256 | 3 | 81.42 | +3.18 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 80.68 | +2.45 |
| Mean | Earlier predictor recipe | 9e-06 | 8 | 1 | 78.27 | +0.03 |
| Mean | Earlier predictor recipe | 3e-05 | 8 | 1 | 78.42 | +0.18 |

### PAWS

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 89.21 | +4.82 |
| CV4 | — | 0.0003 | 256 | 3 | 87.15 | +2.76 |
| Frozen | — | 9e-06 | 128 | 1 | 84.65 | +0.26 |
| Adaptive | — | 9e-05 | 8 | 3 | 84.39 | +0.00 |
| Oracle4 (tuned) | — | 0.0003 | 256 | 3 | 87.30 | +2.92 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 86.42 | +2.04 |
| Mean | Earlier predictor recipe | 9e-05 | 8 | 1 | 83.05 | -1.34 |

### HellaSwag

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 79.88 | +7.03 |
| Frozen + Globalfit | — | 3e-05 | 8 | 3 | 72.92 | +0.07 |
| Adaptive | — | 9e-05 | 8 | 1 | 72.48 | -0.38 |
| Frozen | — | 3e-05 | 8 | 3 | 72.86 | +0.01 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 75.95 | +3.10 |
| Mean | Selected predictor recipe | 3e-05 | 8 | 1 | 73.27 | +0.42 |

### RACE

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 81.19 | +4.95 |
| Adaptive | — | 9e-06 | 8 | 3 | 76.34 | +0.09 |
| Frozen | — | 9e-06 | 8 | 1 | 76.33 | +0.08 |
| Frozen + Globalfit | — | 3e-05 | 32 | 3 | 75.38 | -0.86 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 78.86 | +2.61 |
| Mean | Selected predictor recipe | 9e-06 | 8 | 1 | 76.23 | -0.02 |

### CommonsenseQA

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.001 | 256 | 3 | 78.46 | +4.26 |
| Adaptive | — | 3e-05 | 128 | 3 | 74.17 | -0.03 |
| Frozen | — | 3e-05 | 8 | 1 | 74.04 | -0.16 |
| Mean | Oracle32 recipe | 0.001 | 256 | 1 | 75.27 | +1.06 |
| Mean | Selected predictor recipe | 3e-05 | 128 | 1 | 74.37 | +0.16 |

### SciQ

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 91.57 | +2.97 |
| Adaptive | — | 9e-06 | 8 | 3 | 88.70 | +0.10 |
| Frozen | — | 9e-06 | 32 | 1 | 88.70 | +0.10 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 90.60 | +2.00 |
| Mean | Selected predictor recipe | 9e-06 | 8 | 1 | 88.80 | +0.20 |

### SocialIQA

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 128 | 3 | 72.40 | +3.77 |
| Adaptive | — | 9e-06 | 32 | 3 | 68.41 | -0.22 |
| Frozen | — | 9e-06 | 8 | 1 | 68.47 | -0.15 |
| Mean | Oracle32 recipe | 0.0003 | 128 | 1 | 68.68 | +0.05 |
| Mean | Selected predictor recipe | 9e-06 | 32 | 1 | 68.99 | +0.36 |

### WinoGrande

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 256 | 3 | 65.48 | +6.05 |
| Adaptive | — | 9e-06 | 128 | 3 | 59.83 | +0.39 |
| Frozen | — | 9e-05 | 8 | 1 | 59.59 | +0.16 |
| Mean | Oracle32 recipe | 0.0003 | 256 | 1 | 62.67 | +3.24 |
| Mean | Selected predictor recipe | 9e-06 | 128 | 1 | 59.83 | +0.39 |

### SST-2

| Method | Control settings | LR | Steps | Seeds | Accuracy (%) | Gain over Warmup (pp) |
|---|---|---|---|---|---|---|
| Oracle32 | — | 0.0003 | 128 | 3 | 94.61 | +1.95 |
| Adaptive | — | 9e-05 | 8 | 3 | 92.74 | +0.08 |
| Frozen | — | 3e-05 | 8 | 1 | 92.89 | +0.23 |
| Mean | Oracle32 recipe | 0.0003 | 128 | 1 | 93.23 | +0.57 |
| Mean | Selected predictor recipe | 9e-05 | 8 | 1 | 92.32 | -0.34 |
