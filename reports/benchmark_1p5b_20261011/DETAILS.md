# Experiment Details

[Results overview](README.md) · [Summary CSV](summary.csv)

## True-gradient controls

Positive differences favor the predictor. Improvement over warmup alone does not establish an advantage over these controls.

| Benchmark | Contrast | Difference [95% CI], pp | Update seeds | Matching |
|---|---|---:|---:|---|
| MNLI | CV4 − independently tuned Oracle4 | +1.44 [-0.24, +3.02] | 3 | Independent LR/steps; 4 true-gradient examples/step |
| MNLI | CV4 − matched Oracle4 | +1.29 [-0.61, +3.85] | 3 | Same LR/steps; 4 true-gradient examples/step |
| PAWS | CV4 − independently tuned Oracle4 | -0.15 [-0.34, +0.04] | 3 | Same LR/steps; 4 true-gradient examples/step |
| HellaSwag | Selected predictor − mean-gradient control | -0.29 [-0.55, -0.03] | 1 | Same LR/steps; different true-gradient budgets |
| RACE | Selected predictor − mean-gradient control | +0.06 [-0.16, +0.29] | 1 | Same LR/steps and 20 true-gradient examples/step |
| CommonsenseQA | Selected predictor − mean-gradient control | -0.49 [-1.72, +0.74] | 1 | Same LR/steps and 20 true-gradient examples/step |
| SciQ | Selected predictor − mean-gradient control | -0.10 [-0.30, +0.00] | 1 | Same LR/steps and 20 true-gradient examples/step |
| SocialIQA | Selected predictor − mean-gradient control | -0.72 [-1.44, +0.00] | 1 | Same LR/steps and 20 true-gradient examples/step |
| WinoGrande | Selected predictor − mean-gradient control | -0.32 [-1.74, +1.10] | 1 | Same LR/steps and 20 true-gradient examples/step |
| SST-2 | Selected predictor − mean-gradient control | +0.57 [+0.00, +1.26] | 1 | Same LR/steps and 20 true-gradient examples/step |

The mean-gradient comparison uses one fixed update seed; its interval covers source-group sampling only. CV4 comparisons use three paired seeds. Equal gradient examples per step do not imply equal total gradient work when stopping steps differ, or equal runtime. All intervals here are nominal and conditional on development-selected recipes; family-adjusted intervals remain in `statistics.json`.

Mean paired observed update-time ratios, **CV4 / independently tuned Oracle4**: MNLI 2.35×, PAWS 2.33×. These exclude offline predictor preparation, loading, search and separate gradient audits; they do not establish matched-quality speedups.

## Selected recipes

Each setting is **learning rate × update steps**. Oracle and predictor both start from the listed fixed warmup.

| Benchmark | Warmup | Oracle | Selected predictor | Predictor setting |
|---|---|---|---|---|
| MNLI | 0.001 × 32 | 0.0003 × 256 | CV4 (globalfit) | 0.0003 × 256 |
| PAWS | 0.001 × 32 | 0.0003 × 256 | CV4 (original fit) | 0.0003 × 256 |
| HellaSwag | 0.001 × 32 | 0.0003 × 256 | Frozen (globalfit) | 3e-05 × 8 |
| RACE | 0.0001 × 32 | 0.0003 × 256 | Adaptive | 9e-06 × 8 |
| CommonsenseQA | 0.0005 × 32 | 0.001 × 256 | Adaptive | 3e-05 × 128 |
| SciQ | 0.0001 × 8 | 0.0003 × 256 | Adaptive | 9e-06 × 8 |
| SocialIQA | 0.0005 × 32 | 0.0003 × 128 | Adaptive | 9e-06 × 32 |
| WinoGrande | 0.001 × 32 | 0.0003 × 256 | Adaptive | 9e-06 × 128 |
| SST-2 | 0.001 × 32 | 0.0003 × 128 | Adaptive | 9e-05 × 8 |

## Comparison with the previous 0.5B study

Same full final populations and v4 grading; gains are relative to each model’s own warmup. Recipes and predictor methods can differ, so this does not isolate model size.

| Benchmark | 0.5B oracle gain | 1.5B oracle gain | 0.5B predictor gain | 1.5B predictor gain |
|---|---:|---:|---:|---:|
| MNLI | +8.73 | +6.57 | -0.58 | +4.47 |
| PAWS | +13.18 | +4.82 | +7.01 | +2.76 |
| HellaSwag | +14.14 | +7.03 | +1.45 | +0.07 |
| RACE | +2.61 | +4.95 | +0.41 | +0.09 |

## Protocol and interpretation

- One development-selected warmup per task, seed **201**. Oracle update seeds **101/102/103**; predictor fits **123/124/125**, paired with those update seeds in order. Single-seed unsuccessful screens are explicitly marked.
- LoRA on layer 8 `self_attn.o_proj`, rank = alpha = 64; **196,608 trainable parameters**. FP32 master parameters / BF16 autocast; fresh AdamW, constant LR, weight decay 0, gradient clipping 1, batch size 32.
- Primary predictor fitting uses a Y-only network (width 512, depth 2), 30 epochs, AdamW LR `3e-4`, weight decay `0.01`, and clipping 1. Adaptive calibration uses LR `1e-4` for up to 10 epochs per model-update step, selecting among epochs 0–10 on 16 fixed predictor-development rows. These fitting rates differ from the model-update rates in the recipe table.
- Oracle admission requires a positive three-seed mean over the same warmup on **both development and confirmation**. Predictor LR and stopping are selected independently.
- Globalfit keeps the Y-only architecture, cached fitting data and 30 epochs, while changing both loss normalization and offline checkpoint selection. Its frozen update LR/steps are tuned independently. When the original selected pipeline was adaptive, this also changes online calibration; the accuracy comparison does not isolate a loss-only effect.
- Full evaluation inputs are preserved and verified at generation. Short-answer output generation is capped at 256 new tokens; this is not an input-length limit.
- Grading v4 additionally accepts an entire unambiguous option label such as `A)`. This was fixed before the first oracle search readout; the MNLI warmup choice stayed unchanged. The original readouts are archived. Cross-model accuracy comparisons require harmonized grading.
- Final recipes are frozen before final readout. The prior 0.5B study used the four original benchmark populations, so their cross-model follow-up is exploratory.
- Intervals use paired source-group and update-seed bootstrap, conditional on the fixed warmup. Small positive means and unstable oracle-gain retention ratios should not be described as reliable effects.
- Frozen predictors use no online gradients; Adaptive uses 20 true-gradient examples per step for calibration/validation. Any CV4 follow-up uses four correction examples per step. All methods still use labeled training examples and their full target solutions. These counts measure examples used for true-gradient computation, not annotation cost. Offline training and diagnostic costs remain separate from online updates.
- At most one owned GPU is used at a time. Reservation and monitoring reduce co-tenancy risk; uninterrupted system-level exclusivity is not guaranteed.
- Family-adjusted intervals cover the recorded benchmark family, optionally including registered oracle setup arms, separately for each contrast. They do not correct across every predictor variant, control contrast or adaptive research decision.
- MNLI label_globalfit adds a learned projection of fixed one-hot next-target-token labels to the globalfit predictor. Its label dictionary comes only from predictor_train. It retains the existing labeled data, global loss/epoch metric and 30 fitting epochs, uses no online true gradients, and independently selects update LR/stopping. This architecture supplement is compared with the preceding globalfit cohort; it is not label-free learning.


## Additional limitations

- SST-2 uses sentence identity groups. Original review IDs are unavailable, so training splits are not verified as review-disjoint; the final population is the official labeled validation split.
- Four registered conditional oracle LR extensions were not started. They remain explicitly marked in the machine-readable statistics.
- The nine oracle lower bounds remain positive under the recorded 13-recipe family adjustment, separately for that contrast. This does not correct across every predictor variant or research decision.
- GPU release was verified. Of 41 transient foreign-PID observations, 40 are consistent with owned-worker exits and one remains unresolved. An earlier 56.4-second monitoring gap is recorded; uninterrupted exclusivity is not established.

## Evidence and reproduction

| Artifact | Contents |
|---|---|
| [Statistics](statistics.json) | Paired intervals, retention, controls and unstarted arms |
| [Frozen configurations](frozen_configurations.json) | All 127 variants and checkpoint hashes |
| [Predictor configuration](predictor_configuration.json) | Architecture, fitting and calibration settings |
| [Data sources](benchmark_data_summary.csv) · [Splits](data_split_summary.csv) · [Prompts](prompt_templates.json) | Versions, counts, input hashes and templates |
| [Development results](development_summary.csv) · [LR/step transfer](predictor_hparam_transfer.csv) | Oracle admission and predictor tuning evidence |
| [Costs](cost_accounting.json) · [Selected costs](selected_recipe_costs.csv) · [Four-gradient controls](four_gradient_control_costs.csv) | Measured update times and true-gradient exposures |
| [Task audits](task_audits.json) · [Audit summary](audit_summary.json) | Prediction hashes, inputs, grading and completion checks |
| [Cross-model comparison](cross_model_comparison.json) | Same final populations and harmonized v4 grading |
| [Provenance](provenance.json) · [Checksums](SHA256SUMS) | Original source hashes and publication integrity |

Regenerate the figure with `python render_overview.py` (Matplotlib required). This is a compact publication snapshot, not a standalone training package. Original training artifacts are retained in the local archive.
