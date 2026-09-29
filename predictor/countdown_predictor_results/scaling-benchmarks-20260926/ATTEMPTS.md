# What was tried

[Main results](README.md) · Exploratory changes were motivated by development diagnostics and locked before final outcomes were inspected.

| Attempt | Change | Finding |
|---|---|---|
| Scale and task expansion | Retrain independently for 1.5B/7B x Countdown/GSM8K/MATH | Gradient signals are learnable; no general accuracy advantage |
| Gentle warmup | Reduce warmup LR from 5e-4 to 5e-5; search smaller update LRs | Less starting-point damage; six three-seed Adaptive-versus-Mean intervals still include zero |
| Labels (4 families) | Fixed 128-dimensional random projection of frozen next-token embeddings; both models, original GSM8K/MATH | Added supervision does not reliably improve Adaptive |
| Residual H (2 families) | Replace Y with residual-addition H, same dimensions, labels and fitting budget; original GSM8K | No improvement; 1.5B Adaptive degrades |
| Multistate (2 families) | Gentle GSM8K; split 4096 train / 128 dev across warmup and Oracle steps 8/16/32 | No clear advantage; additionally uses a 1024-example true-gradient teacher trajectory |
| Independent baseline LR tuning | Tune Oracle, Mean and Frozen on generation development data | Only original 7B Countdown Adaptive-minus-tuned-Mean survives primary Holm correction |
| Three-seed replication | Six gentle conditions; shared warmup and splits | No clear Adaptive-minus-Mean or Adaptive-minus-Warmup gain |
| Full MATH / longer Countdown generation | 4999 overlap-free MATH questions; 4096 new output tokens on Countdown | No new general downstream benefit |

## All eight modified families

Accuracy in percent; differences in percentage points. Original Adaptive has its own development-selected LR. Modified Frozen uses its family's Adaptive-selected LR. Intervals are exploratory, unadjusted, conditional on the primary runs.

| Model | Condition | Change | LR | Adaptive | Frozen | Original Adaptive | Adaptive difference [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5B | gsm8k | labels | 0.0001 | 60.42 | 65.13 | 61.11 | -0.68 [-2.58, +1.21] |
| 1.5B | gsm8k | residual_h | 0.0001 | 58.53 | 59.74 | 61.11 | -2.58 [-4.55, -0.61] |
| 1.5B | math | labels | 0.0003 | 36.20 | 23.20 | 35.00 | +1.20 [-2.40, +4.80] |
| 1.5B | gsm8k_gentle | multistate | 3e-06 | 73.01 | 72.71 | 72.71 | +0.30 [-0.99, +1.59] |
| 7B | gsm8k | labels | 0.0003 | 78.47 | 1.59 | 79.23 | -0.76 [-2.88, +1.36] |
| 7B | gsm8k | residual_h | 0.0001 | 77.79 | 72.86 | 79.23 | -1.44 [-3.56, +0.61] |
| 7B | math | labels | 0.0001 | 49.80 | 49.60 | 50.60 | -0.80 [-4.80, +3.20] |
| 7B | gsm8k_gentle | multistate | 1e-05 | 91.74 | 91.66 | 91.58 | +0.15 [-0.68, +0.99] |

The largest local Frozen gain is 1.5B GSM8K with labels: 65.13% versus 61.33% for same-LR original Frozen (+3.79 pp, CI [1.36, 6.22]). Its Adaptive version scores 60.42%. In 7B GSM8K, labels Adaptive beats the original method at the same 3e-4 LR (78.47% vs 76.80%) but trails the original method's selected 1e-4 result (79.23%). Neither example supports a general improvement. Adding labels changes available information; multistate training adds teacher-trajectory supervision.

## Diagnostic findings

- **State sensitivity:** with identical questions, 1.5B Countdown Y cosine remains 0.990 from original warmup to Oracle step 32, while true activation-gradient cosine falls to 0.517. Small feature changes can accompany large gradient changes.
- **Direct task transfer:** hold Countdown weights and predictor fixed, change only the task. LoRA cosine is 0.127/0.044 on GSM8K/MATH for 1.5B and 0.021/0.014 for 7B. This is gradient auditing, not a target-task training experiment.
- **Optimizer effects:** small predicted gradients do not necessarily mean small Adam updates. Original 7B GSM8K step 1 has gradient norm ratio 0.352 but Adam-update norm ratio 0.998; update cosine is only 0.228.
- **Calibration use:** the six gentle primary runs select epoch 0 on 167/192 steps. This retains incoming predictor weights after paying for all 30 trial epochs; it does not skip the LoRA update.
- **Data and numerics:** full MATH contained one predictor-training overlap, excluded in the 4999-question result. Initial nonfinite SDPA-gradient attempts were isolated and recollected with the math backend; failed labels were not used. Three interrupted repeated fits were restarted with the same protocol, not selected for favorable outcomes.

Aggregate/hybrid entries in the original code were not run. Lower calibration frequency, explicit adapter-state inputs and a fresh 20-example-per-step true-gradient baseline remain untested here. This study does not prove that Y + mask + position is the best possible predictor input.
