# What was tried

The completed trajectory/checkpoint inventory is `results/experiment_index.csv`. A configured or queued recipe is not evidence that it completed; use that inventory and the final readouts for results. Development maxima are screening observations, not independent validation.

| Family | Change | Essential comparison / cost |
|---|---|---|
| CV4 / CV8 / CV16 | Predict all 32 examples; correct with 4 / 8 / 16 true gradients | Oracle4 / 8 / 16 and Oracle32, with independently tuned learning rate and stopping point |
| Sparse neural calibration | Fit the predictor head or all weights using already-paid correction labels, after the current update | Fixed predictor; include calibration time |
| Full refresh | Occasionally replace a CV update with the full 32-example true gradient; optionally recalibrate | Charge all refresh examples and fitting time |
| Coefficient calibration | Shrink predictions or estimate separate LoRA-A/B coefficients from previous correction labels | Coefficients are fixed before the current correction sample |
| State conditioning | Add information about the LoRA change since warmup | Same predictor without state features |
| Gradient anchors | Periodically compute a full gradient; reuse its prediction residual between anchors | No-predictor gradient hold on the same schedule; intermediate updates are biased |
| Bernoulli queries | Randomly query a full gradient; otherwise use prediction plus a delayed residual estimate | Matched no-predictor query/EMA schedule; charge actual query count |
| Stratification | Spread correction samples across label strata | Identical sampling without a predictor; reconstruction-only controls |
| Online linear predictor | Fit a small class-conditioned gradient model from past correction labels | No offline neural fit; charge online linear algebra and correction calls |
| Per-example memory | Store and refresh gradients for individual training examples | Charge initial collection and the large memory table; this is not a compact transferable predictor |
| Offline refits | Change width, label features, or centered-residual training objective | Charge gradient-label collection and predictor fitting |
| Residual-stream input | Predict from the layer's residual hidden state rather than only projection output | Original-input refits; gradient-cache numerical differences also change |
| Cached predictor body | Freeze/cache the body's warmup representations; optionally adapt only its head | Current-input predictor; cache and calibration costs |
| Multi-state fitting | Train on gradient labels from several teacher training states | Single-state fit; charge base fit, teacher trajectory, collection, and additional fit |
| Longer trajectories | Follow Oracle/CV checkpoints through 1,024 updates | Compare complete time/quality curves, not just step 32 |

An additional temporal-smoothing grid and the higher-update-learning-rate cached-head variants were deferred before execution to preserve time for paired replications and final validation. A second head-calibration learning rate was also deferred before any cached-head readout. Their prepared code/configurations are not counted as completed experiments. The cached-head screen retains CV4/8/16 at update LR 0.0003, each with a frozen head or head calibration at LR 0.0003.

## Shared execution improvements

Supervised-token-only vocabulary logits avoid materializing logits at positions without loss. Larger microbatches reduce repeated model calls. GPU-resident features remove unnecessary CPU transfers. Frozen-prefix caching skips the unchanged portion of the network. These are applied to applicable Oracle controls as well; their benefits should not be attributed solely to gradient prediction.

The teacher-prefix cache preserves the original residual dtype and recomputes the trainable projection and entire remaining network. Native full input masks and positions remain in use. BF16 batch-shape changes can alter rounding: at steps 32–256, 72/80 saved-weight comparisons were bitwise equal, with differences in MNLI Oracle32. At step 512, 14/20 comparisons matched; MNLI Oracle32/Oracle16/CV16 differed at both tested learning rates. Earlier equality does not guarantee later equality. Non-identical adapters are evaluated separately, and final timing replays must match the specific evaluated weights.

Three-seed standalone CV16 preflights at step 256 matched all PAWS adapters, but only MNLI seed 101; MNLI seeds 102/103 differed. The full-prefix GPU-feature transport implementation remains an alternative for those frozen MNLI checkpoints. Cache speed cannot be paired with another implementation's accuracy merely because a smaller numerical audit passed.

The MNLI CV8 teacher-prefix preflight at step 512 likewise matched seed 101 but differed for seeds 102/103 (relative weight differences 0.082 / 0.257). The frozen original MNLI CV8 and CV16 cohorts therefore use the full-prefix implementation for final timing. The cached MNLI Oracle32 is a separate evaluated cohort. These backend choices were recorded before final readouts.

Y-input refits reuse the historical gradient cache; H-input refits recollect gradients with the optimized backend and larger microbatches. Their comparison is therefore a practical pipeline comparison, not an input-only ablation. The matched-state diagnostic evaluates all fitted predictors against the same fresh gradients, without erasing those differences in fitting data. Multi-state fits also add data and optimization beyond the fixed-H starting predictor; they do not isolate state coverage from every other fitting-budget effect.

## Two different ways to reduce true-gradient calls

**Periodic anchors:** between anchor steps, use the current prediction plus the last observed correction. This is a cheaper, biased approximation. The no-predictor control simply reuses the last true gradient.

**Random queries:** form an estimate `b` before sampling the query indicator `I`, then use `b + I/q × (g − b)`, with query probability `q`. Its expectation is the current true batch gradient before clipping/Adam. The first update is a guaranteed, unscaled true gradient. Reducing query probability can increase variance; unbiasedness does not guarantee useful training.

Neither mechanism is automatically superior to computing a small true-gradient subset every step. Their downstream quality, actual time, and matched controls decide that question.
