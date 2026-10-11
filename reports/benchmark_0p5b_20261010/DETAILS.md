# Detailed Results and Experimental Settings

[Back to overview](README.md)

## All 18 configurations

Accuracy is percent on the predeclared source-disjoint final population. Oracle and predictor columns are means over the listed update seeds. Predictor recipes use three paired fit/update seeds. A dash means no predictor stage was admitted.

| Configuration | N | Seeds | Base | Warmup | Oracle | Predictor |
|---|---:|---:|---:|---:|---:|---:|
| BoolQ | 2713 | 3 | 66.68 | 64.32 | 73.09 | 67.82 |
| MNLI | 9815 | 3 | 39.01 | 71.42 | 80.16 | 70.84 |
| PAWS | 8000 | 3 | 33.98 | 72.67 | 85.86 | 79.69 |
| ARC-Easy | 2376 | 3 | 50.04 | 71.09 | 72.56 | 71.51 |
| OpenBookQA | 500 | 3 | 42.40 | 57.00 | 60.67 | 58.07 |
| HellaSwag | 10042 | 3 | 20.18 | 41.91 | 56.06 | 43.36 |
| RACE | 4934 | 3 | 56.89 | 60.84 | 63.45 | 61.26 |
| ARC-Challenge (primary) | 1172 | 1 | 34.47 | 52.13 | 52.47 | — |
| SVAMP | 300 | 1 | 61.00 | 65.67 | 63.67 | — |
| AQuA | 254 | 1 | 32.28 | 32.28 | 22.44 | — |
| GSM8K (primary) | 1319 | 3 | 48.98 | 47.84 | 42.91 | — |
| Countdown | 2048 | 3 | 4.20 | 8.59 | 12.11 | 7.03 |
| MATH-500 | 500 | 1 | 31.40 | 31.60 | 32.20 | — |
| ARC-Challenge (expanded pool) | 1172 | 3 | 34.47 | 52.13 | 51.42 | 52.16 |
| SVAMP (lower LR) | 300 | 1 | 61.00 | 65.67 | 63.00 | — |
| AQuA (neutral prompt) | 254 | 1 | 34.25 | 37.01 | 33.86 | — |
| GSM8K (lower LR) | 1319 | 3 | 48.98 | 47.84 | 48.24 | 48.27 |
| MATH-500 (lower LR) | 500 | 1 | 31.40 | 31.60 | 33.00 | — |

The five supplemental configurations are expanded-pool ARC-Challenge, lower-LR SVAMP, neutral-prompt/answer-only AQuA, lower-LR GSM8K, and lower-LR MATH. Three other conditional branches were registered but not started because their primary configurations passed. No recipe was retuned after final evaluation.

SVAMP, AQuA, and MATH did not pass development screening; their supplemental searches tied their corresponding warmup on development. Original ARC-Challenge also failed development screening. Original GSM8K passed development but failed confirmation. Single-seed outcomes were not replicated based on their final scores.

## Selected learning rates and steps

Each cell is **learning rate × optimizer steps**. These are selected checkpoints, not necessarily the final step of each search trajectory. Warmup checkpoint selection used development accuracy.

| Configuration | Warmup | Oracle | Predictor |
|---|---|---|---|
| BoolQ | `0.0005 × 8` | `0.0003 × 256` | `9e-05 × 128` |
| MNLI | `0.0005 × 32` | `0.001 × 256` | `3e-05 × 8` |
| PAWS | `0.0005 × 32` | `0.001 × 256` | `0.001 × 128` |
| ARC-Easy | `0.001 × 8` | `0.001 × 8` | `3e-05 × 8` |
| OpenBookQA | `0.001 × 32` | `0.0001 × 256` | `3e-05 × 32` |
| HellaSwag | `0.001 × 32` | `0.001 × 256` | `0.0003 × 8` |
| RACE | `0.0001 × 32` | `0.001 × 128` | `0.0001 × 8` |
| ARC-Challenge (primary) | `0.0005 × 32` | `3e-05 × 8` | — |
| SVAMP | `0.0001 × 32` | `0.0001 × 8` | — |
| AQuA | `0.0005 × 8` | `0.001 × 128` | — |
| GSM8K (primary) | `0.0001 × 8` | `0.0003 × 8` | — |
| Countdown | `0.0001 × 32` | `0.001 × 256` | `3e-05 × 8` |
| MATH-500 | `0.0001 × 8` | `3e-05 × 32` | — |
| ARC-Challenge (expanded pool) | `0.0005 × 32` | `0.0003 × 256` | `9e-06 × 8` |
| SVAMP (lower LR) | `0.0001 × 32` | `3e-06 × 8` | — |
| AQuA (neutral prompt) | `0.0005 × 8` | `1e-05 × 8` | — |
| GSM8K (lower LR) | `0.0001 × 8` | `1e-05 × 32` | `3e-06 × 128` |
| MATH-500 (lower LR) | `0.0001 × 8` | `1e-05 × 1` | — |

## Shared protocol

- Qwen2.5-0.5B-Instruct; LoRA on layer 8 `self_attn.o_proj`; rank = alpha = 64; **114,688 trainable parameters**. The base model is frozen.
- Fresh AdamW for each update trajectory; constant learning rate, zero weight decay, gradient clipping 1; batch size 32; FP32 master parameters with BF16 autocast and SDPA math.
- Warmup grid: LR `{1e-4, 5e-4, 1e-3}`, checkpoints `{8, 32}`, seed 201. Primary oracle grid: LR `{3e-5, 1e-4, 3e-4, 1e-3}`, checkpoints `{8, 32, 128, 256}`, search seed 101; positive candidates replicated with 102/103 before admission.
- Predictor search: oracle LR multiplied by `{0.03, 0.1, 0.3, 1.0}`, checkpoints `{8, 32, 128, 256}`; independent development selection, plus registered CV4 / objective follow-ups.
- Offline predictor: bidirectional Transformer, width 512, depth 2; module output Y, active mask, RMS and position features; raw LoRA A/B gradient supervision; 30 epochs, AdamW LR 3e-4, weight decay 0.01. Offline data normally contain 2,048 training, 128 development, and 128 gradient-test examples; expanded ARC-Challenge uses 512/64/64.
- Oracle admission precedes predictor supervision collection. Admission requires a positive three-seed mean against the same warmup on development and confirmation. It does not require every seed to improve, a minimum two-point gain, improvement over the base model, or a positive confidence-interval lower bound.
- Final recipes and evaluation populations were frozen before final readout. Evaluation uses greedy answer generation, not standard likelihood-ranking benchmark protocols.

## What the predictor methods do

| Method | Online true-gradient examples per step | Use |
|---|---:|---|
| Frozen | 0 | Apply the offline predictor without online calibration |
| Adaptive | 20 | Four current examples and 16 fixed predictor-development examples for calibration / validation; their true gradients are not directly mixed into the LoRA update |
| CV4 | 4 | Correct a predicted batch gradient using four uniformly sampled true-gradient examples |

CV4 is a control-variate estimator. For a batch of 32, it uses `(sum(predicted gradients) + 8 × sum_4(true − predicted gradients)) / supervised_token_count`. Conditional unbiasedness applies before clipping and AdamW, not to downstream accuracy. Offline supervision costs still count for all predictor methods.

Online calibration often selected epoch zero: 12 of 15 selected adaptive trajectories accepted no predictor training epochs at any step. The selected GSM8K adaptive runs all did so. Their gains therefore do not demonstrate successful online predictor learning.

The mean-gradient diagnostic directly updates LoRA with the same 20 true-gradient examples at matched learning rate and step count; it uses update seed 101 only. It matches Adaptive's online example budget, but not Frozen's zero-online-gradient budget. BoolQ CV4 has no demonstrated advantage over its matched four-example oracle. PAWS CV4 is below an independently tuned four-example oracle. The evidence does not establish a general predictor advantage over these controls.

## Supplemental recipe changes

- **ARC-Challenge:** update pool expanded from 256 to 768 by adding 512 predictor-training questions. This declared overlap is within training data; confirmation and final source groups remain excluded. Its positive development/confirmation means did not carry through to a positive final oracle mean.
- **GSM8K:** lower learning rate and changed stopping point improve oracle mean accuracy by **5.33 pp**, with paired nominal 95% interval **[2.60, 7.91]** versus the primary recipe. Both use the same fixed warmup and update seeds. Supplemental confirmation reuses the primary confirmation rows; it is not a fresh independent replication. The final oracle gain over warmup is only **+0.40 pp**, with interval **[−0.91, 1.74]**.
- **AQuA:** neutral prompting and answer-only oracle training reuse the original warmup weights, reevaluated with the new prompt. The resulting warmup accuracy differs because of the prompt, not a new warmup seed or new warmup training.
- **OpenBookQA:** global-fit supervision uses fixed training-set factor-energy denominators and a changed checkpoint criterion; the predictor LR was retuned. This combined change improves final predictor accuracy, but does not isolate one causal component.
- **BoolQ / PAWS:** CV4 follow-ups were selected using development results. The final selected methods retain some oracle gains, but require true-gradient corrections.

## Evaluation and uncertainty

All original final inputs were evaluated without truncation. Primary statistics exclude final source groups seen in any non-final split. This leaves **2,713 of 3,270 BoolQ examples**, grouped into 2,541 source groups. The other completed configurations retain their full final sets. Both full-set and primary scores are in `final_variants.csv`. BoolQ development has 17/128 examples sharing passages with non-final training/calibration/diagnostic sources, although the questions differ; the independent confirmation and final source exclusions are retained.

MNLI uses matched data; PAWS uses Wiki labeled_final; HellaSwag uses validation; RACE combines levels; Countdown uses a fixed 2,048-example test. MATH trains on MATH and evaluates on MATH-500.

Gain intervals use 10,000 paired bootstrap replicates over source groups and update seeds, conditional on the fixed warmup. Plots and summary CSV use nominal 95% intervals. `task_audits.json` also retains family-adjusted intervals. Single-seed results do not measure update-seed variability. Small positive means, including ARC-Easy, OpenBookQA, RACE, and lower-LR GSM8K predictors, have intervals including zero.

Retention ratios are descriptive and unstable when the oracle gain is near zero or its interval includes zero. In particular, do not interpret GSM8K's ratio as reliably exceeding 100%; ARC-Challenge has no positive final oracle gain to retain.

## Execution and release boundary

The study finished in approximately **13.9 hours** from its configured start through final delivery checks, within the 16-hour budget. The recorded allocation used at most one owned GPU at a time. An early GPU co-tenancy incident and a later **2.36-minute monitoring gap** are documented; uninterrupted exclusive GPU timing is not claimed. GPU resources were released after completion.

The local study audited 18 configurations, 156 variants, and 591,561 raw predictions, including full-prompt checks and grading consistency under the frozen rules. This repository release preserves compact statistics, per-variant results, selected settings, audit records, and source hashes. It supports inspecting results and regenerating the overview chart. Full training or grading reproduction additionally requires the original datasets, raw generations, weights, caches, and experiment implementation; these are not bundled here.
