# Source snapshots

Snapshots of the executed single-layer code; only text line endings and final blank lines are normalized. `update_adaptive.py` implements calibration, validation selection, and LoRA updates; `multilayer.py` implements gradient collection and diagnostics; `predictor.py` contains the architecture and A/B reconstruction; `fit_ablation.py` is the offline trainer; `eval_accuracy.py` and `core.py` contain evaluation and shared setup.

These references require the original experiment layout, model weights, data, and shared Countdown helpers; they are not a standalone reproduction package. The scripts also contain all-layer branches that were not completed in this experiment. See `../protocol.json` for the executed single-layer settings. Large checkpoints, caches, and full generated outputs remain on the server.
