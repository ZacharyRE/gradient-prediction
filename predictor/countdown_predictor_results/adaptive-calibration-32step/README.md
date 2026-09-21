# Adaptive predictor calibration: 32 LoRA updates

**Four calibration examples per batch, plus 16 fixed gradient-validation examples, reach 7.32% test accuracy after 32 updates — baseline 4.49%, oracle 7.45%.**

Qwen2.5-0.5B-Instruct · layer 8 `o_proj` only · LoRA rank/alpha 64 · teacher32 starting adapter · selected `Y + mask + position` predictor.

## What happens at each step?

1. Take a batch of **32 examples**; randomly select **4** to calibrate the predictor using true gradients.
2. Recompute true gradients on **16 fixed, separate development examples** at the current LoRA state. These select the checkpoint and do not train the predictor.
3. Train for **30 epochs** on the four examples (one optimizer step per epoch). Evaluate every epoch, including the incoming **epoch 0**; restore the lowest-error checkpoint.
4. Predict gradients for **all 32 examples**, aggregate by supervised-token count, and take **one LoRA update**. Repeat through step 32.

Selection score: **per-example A/B relative squared error + aggregated A/B relative squared error**. Downstream accuracy is not used to choose the calibration epoch. Across 96 refreshes, the selected epoch averages **6.45** (median **3**); epoch 0 is selected 8 times and epoch 30 four times.

## Complete test results

**2,048 questions for every checkpoint; means over three paired seeds.** Starting accuracy: **92/2048 = 4.49%**. Test endpoints 1, 16, and 32 were specified in advance.

| Method | Step 1 | Step 16 | Step 32 |
|---|---:|---:|---:|
| Validation-selected calibration epochs | 5.18% | 4.87% | **7.32%** |
| Fixed 5 calibration epochs | 5.00% | **6.22%** | 6.61% |
| Frozen predictor | 5.18% | 0.29% | 0.00% |
| True-gradient oracle | 5.11% | 5.71% | **7.45%** |

![Accuracy across update steps](figures/accuracy.png)

At step 32, adaptive calibration improves over baseline by **2.83 percentage points**, exceeds fixed-5 by **0.72 points**, and trails oracle by **0.13 points**. Its per-seed accuracies are **6.79%, 7.86%, 7.32%**; all three exceed baseline and their paired fixed-5 control. At step 16, fixed-5 is better. Performance is not monotonic, and proximity to oracle does not establish equivalence.

## Configuration

| Setting | Value |
|---|---|
| Predictor | Bidirectional Transformer; width 512, 2 layers, 8 heads, FFN 1024; 5.13M parameters |
| Initial predictor training | 8,192 train / 256 gradient-dev examples; 100 epochs; batch 32; AdamW lr **3e-4** |
| Calibration | **4 current-batch train + 16 fixed gradient-dev**; AdamW lr **1e-4**, weight decay 0.01, clip 1 |
| Calibration training loss | Individual A/B relative MSE + 0.25 normalized activation-gradient MSE; no aggregate training term |
| Epoch selection | Minimum gradient-dev score among **0–30**; run all 30, then restore best |
| LoRA update | AdamW lr **3e-4**, weight decay 0, clip 1; batch 32; 32 steps |
| Predictor/update seeds | **123/101, 124/102, 125/103** |
| Generation | Greedy BF16 vLLM; complete inputs; **2,048 new-token output cap** |

**Comparison caveat:** adaptive resets predictor AdamW at each batch; fixed-5 retains its optimizer state. Both retain predictor weights, and both retain LoRA optimizer state. Their difference therefore cannot be attributed solely to epoch selection. All three adaptive runs select epoch 0 at the first step. The remaining 28 current-batch true gradients are diagnostic only; their acquisition adds measurement cost. The 16 validation examples also require fresh true gradients each step.

These are completed **single-layer** results on the existing test benchmark. All-layer downstream experiments remain unfinished.

[Full configuration](protocol.json) · [Per-seed test results](results/test_accuracy.csv) · [Accuracy-dev results](results/dev_accuracy.csv) · [All epoch choices and sampled indices](results/calibration_choices.csv) · [Gradient metrics](results/gradient_metrics.csv) · [Audit](results/test_audit.json) · [Source snapshots](source/) · [SHA256 provenance](provenance.json)
