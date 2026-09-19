# Implementation references

These are unchanged source snapshots, included to inspect the architecture, losses, calibration loop, and gradient reconstruction. They are not a standalone runnable package: execution requires the original experiment directory layout, Qwen weights, cached gradient supervision, and shared Countdown helpers. Local filesystem paths in these snapshots reflect the original environment.

- `current/predictor.py`: selected bidirectional Transformer and auxiliary inputs.
- `current/fit_ablation.py`: offline predictor training and checkpoint selection.
- `current/update.py`: frozen one-step LoRA update.
- `current/update_calibrated.py`: new 4-of-32 online calibration experiment.
- `countdown2/dynamic16_online.py`: earlier fixed-probe calibration and optional hybrid estimator.
- `countdown2/dynamic16_mean.py`: true-gradient mean control, distinct from a neural predictor.

Large weights, caches, datasets, and generated answers are not included. Checkpoint paths and SHA256 identities are in `../configs/checkpoint_manifest.json`. JSON paths are normalized relative to the original project root; normalization does not change numerical results.
