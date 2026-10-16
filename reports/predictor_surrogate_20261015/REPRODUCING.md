# Reproducing the evidence

This repository contains recipes, source code, compact result tables and provenance. Large base-model weights, warmup adapters, learned predictor checkpoints and raw generated responses remain in the original experiment storage. A checkpoint hash identifies an artifact; it does not replace that artifact. Exact GPU reproduction additionally depends on the recorded software/hardware environment.

## Reconstruct the input datasets

`provenance/data_reproduction.json.gz` records the source parquet hash, selected original row indices and formatting metadata for all 17 used MNLI splits. The reconstruction audit checked byte-identical JSONL files, preserving complete evaluation problems. Generation separately verifies the full prompt token IDs. Obtain the matching MNLI training parquet and run the published `scripts/data_reproduction.py` with `--help` for its `--metadata`, `--raw-dir` and `--output-dir` options. It refuses a different source hash or an existing conflicting output.

The selection-dev and selection-holdout populations are disjoint premise groups sampled from MNLI training data; they are not the official MNLI validation set. The metadata artifact contains no generated model outcomes.

Exact source: [`nyu-mll/multi_nli`, revision `da70db2af9d09693783c3320c4249840212ee221`](https://huggingface.co/datasets/nyu-mll/multi_nli/resolve/da70db2af9d09693783c3320c4249840212ee221/data/train-00000-of-00001.parquet). Save the linked file as `train-00000-of-00001.parquet` in a raw-data directory. Its SHA256 must be `1c1de03640b168e410aabfca19e7cc2f3dfcd7f0e126e935674e56fb102c4529`.

```bash
python predictor/surrogate_reuse_15h_20261015/scripts/data_reproduction.py \
  --metadata reports/predictor_surrogate_20261015/provenance/data_reproduction.json.gz \
  --raw-dir /path/to/raw-mnli \
  --output-dir /path/to/reconstructed/mnli
```

Dataset reconstruction uses NumPy 2.3.5 and PyArrow 25.0.1; compact statistical reproduction needs NumPy. Use the recorded NumPy version for exact numerical equality. GPU replay uses the full recorded environment in `provenance/environment.json`.


## Replay one selected training recipe

`results/selected_training_receipts.json` connects every final adapter to its exact training configuration and hashes. From the repository root, run:

```bash
python predictor/surrogate_reuse_15h_20261015/scripts/reproduce_surrogate_recipe.py \
  --receipts /path/to/selected_training_receipts.json \
  --tag RECORDED_ENTRY_TAG \
  --warmup-adapter /path/to/matching/warmup \
  --predictor-checkpoint /path/to/matching/best.pt \
  --data-dir /path/to/reconstructed/mnli \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --output /path/to/new/replay \
  --gpu 0 --prepare-only
```

Omit `--predictor-checkpoint` for a recipe that has no predictor dependency. Preparation validates warmup, predictor and training-data hashes and writes an isolated command without using a GPU. To execute, choose another fresh output directory and omit `--prepare-only`. The runner acquires the cooperative GPU admission lock, refuses an occupied card at startup and enforces its own duration limit. It does not preempt other workloads. The selected recipes use `reuse_train`; the earlier tangent-space exploratory trainer has a separate entry point.

The replay preserves the recorded learning rate, optimizer, sample sequence, full inputs, stopping point and backend. Disabling intermediate diagnostics does not change updates. Scheduled recipes may only be replayed at their original horizon. Serialized adapter hashes are reported; the study's timing qualification uses the stronger direct tensor-by-tensor equality check in `replay_reuse.py`.

The `--prepare-only` relocation path was checked on CPU. A successful preparation is not presented as a new GPU reproduction run. Final speed evidence comes from the separately recorded standalone timing/FLOP replays.

## Reproduce statistical summaries

The final freeze fixes recipes, adapters, evaluation code and analysis specification before any candidate-selection readout. The second freeze fixes choices using selection-dev before selection-holdout evaluation.

The compressed `frozen_correctness.npz` bundle stores pickle-free per-example correctness for every frozen adapter, together with sample IDs, counts and data/freeze/decision hashes. From the repository root, after reconstructing the input data, run:

```bash
python predictor/surrogate_reuse_15h_20261015/scripts/reproduce_surrogate_statistics.py \
  --study-dir predictor/surrogate_reuse_15h_20261015 \
  --report-dir reports/predictor_surrogate_20261015 \
  --data-dir /path/to/reconstructed/mnli \
  --output /path/to/new/statistics-replay
```

This CPU-only command validates frozen source/data/bundle hashes, builds a fresh experiment layout and reruns the published analysis. It checks exact equality of every numerical quality, ranking, regret and paired-bootstrap field against the published result. No model weights, GPU or original absolute experiment paths are needed. See `results/surrogate_analysis_reproduction_audit.json` for the actual replay outcome. This reproduces statistics; it does not independently regrade responses, because the compact bundle excludes generated text.

The same command also passed when invoked from the isolated publication worktree using its copied public files; see [publication-layout replay](provenance/publication_layout_statistics_reproduction.json).

The reported intervals resample the three paired warmup/update-seed starts and source-premise groups, conditional on the already frozen candidate choices. Three starts provide limited evidence. Reported intervals are descriptive and are not adjusted for the many method comparisons. Candidate subsets are fixed experimental choices, not additional independent seeds. The best-holdout-candidate ceiling is descriptive and cannot be deployed without accessing holdout outcomes.

## Source and timing boundaries

Every live stage executed an immutable source snapshot, with per-file hashes retained locally. Published current source includes the final analysis and replay tools; historical exploratory code revisions are identified by their receipts. Operational process logs and foreign-process information are excluded from publication.

Development timers and historical predictor-preparation components are not clean cold-start benchmarks. Standalone speed tables require three distinct fresh-process repetitions with exact final weights. FLOP-instrumented timings are excluded from speed statistics. Matrix/convolution FLOPs omit scalar and nonlinear operations, and offline FLOPs are reported only when actually measured.

## Complete candidate-selection workflow timing

The registered workflow benchmark uses heldout start 302 and the five random candidates, with three fresh repetitions of six fixed simulator/readout combinations. It reruns five candidate adaptations, one selection-dev readout and, when needed, one final selected reference-Oracle adaptation. The exhaustive reference-Oracle workflow reuses its already-trained chosen adapter. Model loading, subprocess transitions, output writing and verification are inside this timer; source predictor preparation and the common target warmup are outside it.

Each fresh adaptation must reproduce the frozen weights tensor by tensor. To retain the exact frozen evaluation manifest, the scorer opens the tensor-equivalent reference artifact paths; it recomputes every prediction and checks them against the frozen records. Candidate choice must match exactly. The teacher-only scorer omits learned-behavior inference and initial-anchor construction. `selection_pipeline_summary.json` records completion and repeated timing qualification; incomplete conditions are not speed claims. Three timing repetitions are one scientific starting state, not three seed replications. Other candidate families retain explicitly labeled component-cost estimates.

The [split-integrity audit](results/split_integrity_audit.json) checks the actual 17 selected input files: source groups are disjoint except for the deliberately overlapping random candidates and their original update pool. It also verifies exact registered candidate membership and exclusion of previous-study groups from the new selection and genre populations. This check makes no claim about base-model pretraining contamination.

Post-validation completion policies apply only to pinned, already registered development jobs after mandatory scientific validation passed. Their wrappers change the wall-clock reserve while executing the unchanged training implementation; the actual user deadline and GPU cutoff remain enforced. These controls remain exploratory and never modify the original final freeze. Public provenance omits operational process identifiers and retains hashes of the unmodified local receipts.
