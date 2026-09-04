# Hidden-State Gradient Predictors

This directory is the organized entry point for predictor code and results.

## Layout

- `single_layer/README.md`: detailed predictor protocol and local result layout.
- Canonical implementations live in the repository-level `training/`, `evaluation/`, and
  `gradient_geometry/` directories.
- Canonical experiment definitions live in `configs/gradient_geometry/`.

Local `training_code/` compatibility links and `results/` artifacts are intentionally excluded from
Git because the links contain machine-specific absolute paths and the raw artifacts are large.
Maintained result tables live in `docs/RESULTS.md`.

## Shared conventions

- Model: Qwen2.5-1.5B-Instruct.
- Predictor train / held-out test: 2000 / 1000 MATH-train examples, seed 42.
- LoRA warmup examples are disjoint from predictor train/test.
- Per-example gradients use assistant-solution-only supervised loss.
