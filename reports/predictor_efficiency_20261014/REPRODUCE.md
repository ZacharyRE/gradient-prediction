# Reproduce a trajectory

The repository contains code, exact recipe settings, artifact hashes, and result tables. Model weights and dataset rows remain local. Use the recorded study data, task warmup adapter, and predictor checkpoint; a fresh repository clone alone does not contain those inputs. The package versions are in `provenance/environment.json`.

Complete historical per-run source bundles remain in the local study archive. The published trainers implement the final replay paths; their replay receipts verify the evaluated weights. The inventory also retains source hashes for earlier screens.

From the repository root, this example replays the frozen PAWS CV16 recipe and evaluates every saved checkpoint plus the shared warmup on `accuracy_dev`:

```bash
python predictor/cv4_effectiveness_9h_20261014/scripts/reproduce_selected.py \
  --source-study /path/to/benchmark_1p5b_11h_20261011 \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --jobs predictor/cv4_effectiveness_9h_20261014/configs/frozen_paws_cv16_u101_jobs.json \
  --recipe fast_cv16_cache \
  --module research_teacher_cache \
  --task paws --seed 101 --predictor-seed 123 \
  --output /path/to/new_replay --gpu 0 --steps 256 --evaluate
```

Use a **new output directory** and an otherwise vacant chosen GPU. `--prepare-only` creates the replay layout and commands without training. `--warmup-adapter` accepts a relocated copy of the recorded warmup weights and checks its hash. `--predictor-checkpoint` supplies a relocated or newly fitted predictor. The optional `--duration-minutes` defaults to 60, with a 10-minute cleanup reserve.

`--compute-cutoff-epoch` can additionally impose an absolute Unix-time cutoff for compute. It only shortens the duration budget; training and evaluation child processes share that bounded window.

The launcher holds the same cooperative per-GPU admission lock used by this study across training and evaluation. GPU occupancy is checked before each model stage, and the worker reserves unused allocator memory; this does not provide administrator-enforced exclusive access.

The launcher preserves full evaluation inputs. Its 256-token setting caps **newly generated output**, not the prompt. It evaluates the development split only; frozen confirmation/test readouts use the recorded manifest and the main study evaluator.

For other methods, choose the corresponding job name and trainer module from the published configurations and selected timing receipts. The final frozen manifest records update seed, predictor seed, learning rate, stopping step, adapter hash, and comparison cohort. Online-only linear predictors use predictor seed zero. New offline refits require their own fitted checkpoint rather than the inherited neural predictor.

Timing replays use `time_recipe.py`; matrix-FLOP replays use `profile_recipe.py`. Both consume the same recipe. FLOP instrumentation changes execution time, so its wall time must not be reported as training speed. Standalone timing receipts compare the replay's final weights with the evaluated adapter and separately record update time, preparation, and process-level script time.

Those two study-harness scripts expect the reservation/deadline configuration used by the main queue. The launcher above is the supported entry point for a fresh single-trajectory replay; it creates a fresh deadline and holds its own cooperative lock. It does not silently rerun the entire nine-hour selection sweep or the frozen final tests.

**Validated:** the fresh-directory launcher completed PAWS CV16 through 32 updates, followed by all 512 development examples for the resulting adapter and shared warmup. The trained weights are bitwise identical to the original step-32 checkpoint; both generated accuracies also match (CV16: 87.50%, warmup: 84.18%). See the [smoke-test receipt](provenance/reproduction_smoke.json). This is a reproduction check, not another speed measurement or new final selection. All 48 final standalone timing replays separately match their evaluated weights.
