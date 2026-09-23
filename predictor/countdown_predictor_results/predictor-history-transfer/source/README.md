# Source snapshots and reproduction

`experiment/` contains the unchanged scripts from the completed local experiment:

| Script | Role |
|---|---|
| `prepare.py` | Select disjoint new data and freeze source/checkpoint identities |
| `run_seed.py` | Matched history calibration, gradient diagnostics, and actual one-step updates |
| `eval_generation.py` | Prespecified complete-input accuracy evaluation |
| `low_lr_control.py` | Outcome-informed, loss-only learning-rate diagnostic |
| `analyze.py` | Audit and aggregate paired results and uncertainty |

`dependencies/` preserves the predictor/model helpers and Countdown data,
prompt and verifier helpers imported by those scripts. Original paths and
SHA256 identities are recorded in [provenance.json](../provenance.json).

These are research snapshots, **not a standalone training package**. To rerun,
restore the scripts to `predictor/predictor_history_transfer_20260923/` in the
original experiment layout, and restore helper files to the original repository
paths listed in provenance. The original Countdown source and prior data
partitions, Qwen model snapshot, three pretrained predictors and oracle LoRA
checkpoints are required. Their identities and source indices are recorded in
the [data manifest](../data_manifest.json). Repository-root prefixes in that
published manifest are normalized; preparation regenerates the native local
manifest used by the scripts.

From the original `predictor/` directory, using the recorded environment:

```bash
python predictor_history_transfer_20260923/prepare.py
python predictor_history_transfer_20260923/run_seed.py --seed 101
python predictor_history_transfer_20260923/eval_generation.py --seed 101
```

Repeat the last two commands for seeds 102 and 103. Each main seed requires
a new output directory. Run `low_lr_control.py` after all main seeds, then
`analyze.py --require-accuracy`. Put the environment's `bin` directory on PATH
for vLLM's ninja dependency. [Environment versions](../environment.json).

Weights, datasets, gradient caches, detailed per-token losses and raw generated
answers remain in the original server archive. The release includes compact
paired metrics, all 1,024 per-question correctness vectors and hashes of the
archived records. The [completion audit](../results/audit.json) refers to those
original artifacts; it does not imply their files are bundled here.

The overview figure can be regenerated using only this release's compact CSVs:

```bash
python source/plot_overview.py
```
