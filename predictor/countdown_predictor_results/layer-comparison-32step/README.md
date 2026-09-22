# Layer Comparison: Configuration and Evidence

**[Main report: joint and independent layer updates →](../README.md#joint-and-independent-layer-updates)**

Completed 2026-09-21. Independent layer 3 and 19 runs supplement the original layer-8 reference; a separate run jointly updates layers 3/8/19. Layer numbers are zero-indexed. Each scope has its own 32-step real-gradient warmup.

| Evidence | Contents |
|---|---|
| [Joint configuration](configs/joint/experiment.json) / [Independent configuration](configs/independent/experiment.json) | Executed warmup, offline training, calibration, update, seed, and evaluation settings |
| [Joint model protocol](configs/joint/protocol.json) / [Independent model protocol](configs/independent/protocol.json) | Exact base-model snapshot, LoRA rank/alpha, update scope; original local paths retained |
| [Run configurations](configs/run_configs.json) | Recorded warmup, offline predictor, and update settings per method/seed |
| [Per-seed CSV](results/accuracy.csv) / [Mean CSV](results/mean_accuracy.csv) / [JSON](results/summary.json) | All 440 dev/test evaluations across four scopes, including 148 test evaluations; warmup uses step 0, meaning zero **post-warmup** updates; accuracy is a fraction |
| [Offline training](results/offline_training.csv) | Selected epochs and scores for all 15 newly fitted predictors; each completed 100 epochs. The [original layer-8 protocol](../adaptive-calibration-32step/protocol.json) covers reused fits |
| [Calibration choices](results/calibration_choices.csv) | 480 per-layer adaptive refreshes from the new joint and independent runs; selected epochs, scores, and calibration indices. Original layer 8 is in its [existing bundle](../adaptive-calibration-32step/results/calibration_choices.csv) |
| [Joint gradient metrics](results/joint_gradient_metrics.csv) / [Independent gradient metrics](results/independent_gradient_metrics.csv) | Every step's activation, per-example LoRA, and batch LoRA cosine, relative L2, and norm ratio; independent CSV includes layer 8 |
| [Independent gradient plot](figures/independent_gradients.png) / [PDF](figures/independent_gradients.pdf) | Adaptive full-batch metrics, mean ± sample SD over three seeds; trajectories have different model states |
| [Evaluation integrity](audits/evaluation_integrity.json) / [Run audits](audits/run_audits.json) | Counts, correctness, input order, unchanged base model; additional input hashes and warmup/update audits in [audits/](audits/) |
| [Source snapshots](source/README.md) / [Provenance](provenance.json) | Executed source/config identities and reproduction boundary |

Each evaluation preserves the full prompt. Dev has 256 questions, test has 2,048; these are distinct from the 256-example predictor-dev set used for offline selection and its fixed 16-example online subset. The test benchmark has been used before; no new independent confirmation set is claimed.

Gradient diagnostics compare predicted and true gradients at the **same current model state and batch**. Relative L2 is `||predicted − true|| / ||true||`; norm ratio is `||predicted|| / ||true||`. `heldout_before/after` use the 28 current-batch examples excluded from calibration training; `full_batch` uses all 32. True diagnostic gradients add computation. Oracle has no predictor reconstruction metric.

These are compact evidence and source archives. Model checkpoints, datasets, raw generated answers, logs, and gradient caches remain on the server. Configuration paths and original hash-manifest paths describe that server layout, not downloadable GitHub assets.
