# Executed source snapshots

[Joint three-layer pipeline](joint/run_pipeline.py) · [Independent layer-3/19 pipeline](independent/run_pipeline.py) · [Original layer-8 source](../../adaptive-calibration-32step/source/README.md)

The `joint/` and `independent/` files are copies of the executed scripts with only trailing whitespace, line endings, and final blank lines normalized; Python syntax trees are unchanged. They include warmup, gradient collection, predictor architecture/training, calibration, LoRA updates, and full-input evaluation. File identities and original locations are recorded in [provenance](../provenance.json).

These snapshots are for inspection, not standalone runnable packages: relative paths assume the original `predictor/predictor_dynamic/<experiment>/scripts/` layout and corresponding configs, data, model snapshots, initialization adapters, and provenance files. Shared `common`, `sft_helpers`, and `countdown` modules come from the original Countdown research tree. Restore those dependencies and the original layout before running a pipeline. Training was restricted to physical GPU 0/1; no training is triggered by publication.

The snapshot evaluation scripts preserve full inputs. The 2,048-token option caps generated output only. Original source hashes describe execution; later report/monitor scripts are not part of training.
