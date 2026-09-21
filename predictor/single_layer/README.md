# Earlier Single-Layer Gradient Prediction

[Predictor research map](../README.md) · [Historical results](../../docs/DATA_SELECTION_RESULTS.md) · [Original experiment plan](../../docs/DATA_SELECTION_PLAN.md)

This is the Qwen2.5-1.5B / MATH–GSM8K static-gradient and data-selection pipeline. It is separate from the current Countdown calibration study. Commands below run from the repository root; local result arrays, model weights, and compatibility symlinks are not distributed in Git.

These experiments predict the complete 12,288-dimensional raw rank-4 LoRA A+B gradient for layer 17
without CountSketch or SVD.

## Training code

- [training/warmup_lora.py](../../training/warmup_lora.py): independent LoRA warmup.
- [evaluation/extract_single_layer_raw_gradients.py](../../evaluation/extract_single_layer_raw_gradients.py): paired hidden-state and raw-gradient extraction.
- [evaluation/evaluate_direct_raw_gradient_ridge.py](../../evaluation/evaluate_direct_raw_gradient_ridge.py): train-only Ridge CV and final fit.
- [evaluation/evaluate_direct_raw_gradient_factor_ridge.py](../../evaluation/evaluate_direct_raw_gradient_factor_ridge.py): joint CV over separate
  LoRA A/B Ridge penalties.
- [evaluation/evaluate_direct_raw_gradient_mlp.py](../../evaluation/evaluate_direct_raw_gradient_mlp.py): MLP with train-only early stopping.
- [evaluation/evaluate_structured_lora_gradient_bottleneck.py](../../evaluation/evaluate_structured_lora_gradient_bottleneck.py): shared bottleneck with separate,
  independently normalized LoRA A/B heads while retaining the flattened label files.
- [evaluation/evaluate_low_rank_ridge_bottleneck.py](../../evaluation/evaluate_low_rank_ridge_bottleneck.py): pure-linear reduced-rank Ridge with a shared
  64-dimensional bottleneck and separate flattened A/B output heads.
- [evaluation/evaluate_separate_low_rank_ridge_bottlenecks.py](../../evaluation/evaluate_separate_low_rank_ridge_bottlenecks.py): independently solved pure-linear
  Ridge and rank-64 bottlenecks for A and B, with no shared encoder.
- [evaluation/evaluate_target_alignment_spearman.py](../../evaluation/evaluate_target_alignment_spearman.py): extract the 500-example target
  validation gradient and measure candidate target-alignment ranking Spearman.
- [configs/gradient_geometry/](../../configs/gradient_geometry/): layer-17 `o_proj`, `q_proj`, `v_proj`, and `down_proj` configurations.
- [evaluation/run_qvo_layer_rank_sweep.py](../../evaluation/run_qvo_layer_rank_sweep.py): resumable three-GPU sweep over layers 5, 11, 23,
  and 27 for `q_proj`, `v_proj`, and `o_proj` at ranks 2 and 4.
- [evaluation/summarize_qvo_layer_rank_sweep.py](../../evaluation/summarize_qvo_layer_rank_sweep.py): generates the cross-layer/rank JSON and
  Markdown comparison tables.
- [evaluation/summarize_hidden_pooling_comparison.py](../../evaluation/summarize_hidden_pooling_comparison.py): paired comparison of last-token and
  mean-pooled prompt hidden states for layers 5 and 11.
- [evaluation/cache_prompt_hidden_sequences.py](../../evaluation/cache_prompt_hidden_sequences.py): cache one ragged target-module-input prompt
  tensor per sample and verify that every cached `H[-1]` matches the existing last-token feature.
- [evaluation/extract_alternative_single_layer_hidden.py](../../evaluation/extract_alternative_single_layer_hidden.py): matched hidden-source ablations that
  reuse the fixed adapter, splits, raw gradients, and target gradients while recomputing only the
  prompt representation.
- [evaluation/evaluate_hidden_representation_ablation.py](../../evaluation/evaluate_hidden_representation_ablation.py): strictly matched cosine-loss ablation of
  last-token, mean-pooled, and learned second-order prompt representations with joint and
  factor-specific LoRA-gradient objectives.

## Local artifact layout

The paths in this section describe the server archive under `predictor/single_layer/results/`. The public numerical summary is in [historical results](../../docs/DATA_SELECTION_RESULTS.md).

- `layer17_oproj_rank4_direct/`: block-hidden-state to layer-17 `o_proj` gradient.
- `layer17_qproj_rank4_direct/`: exact layer-17 `q_proj` module input to its gradient.
- `layer17_vproj_rank4_direct/`: exact layer-17 `v_proj` module input to its gradient.
- `layer17_downproj_rank4_direct/`: exact layer-17 `down_proj` module input to its gradient.
- `layer{5,11,23,27}_{q,v,o}proj_rank{2,4}_direct/`: 24 cross-layer/rank experiments using
  the same fixed 2,000/1,000 split and warmup protocol.
- `qvo_layer_rank_summary.{json,md}`: unified metrics for the 24 new experiments plus the
  existing layer-17 rank-4 reference experiments.
- `hidden_pooling_comparison.{json,md}`: paired last-token versus mean-pooling results.
- `hidden_source_matched_comparison.md`: exact O-module-input and post-layer Q/V comparisons for
  layers 5 and 11 at ranks 2 and 4.

Within each experiment, `warmup_128_steps32/` stores the adapter and warmup report;
`formal_2000_1000/` stores extraction arrays; its `ridge/` and `mlp_width512/`
subdirectories store the corresponding trained predictors and summaries. `factor_ridge/` stores the
separate-alpha A/B Ridge predictor and its full pairwise CV grid.
`formal_5000_1000/` extends the same 2,000-example training split with 3,000 non-overlapping rows
while preserving the exact original 1,000-example candidate test set.
`structured_bottleneck_width64/` stores the shared-width-64 A/B-head predictor.
`ridge_bottleneck_rank64/` stores the pure-linear rank-64 Ridge bottleneck predictor.
`separate_ridge_bottlenecks_rank64_64/` stores independent rank-64 Ridge predictors for A and B.
`target_alignment/` stores target raw gradients, per-candidate alignment arrays, and the Spearman
summary. The same metric is also inserted into each predictor's own summary JSON.

## Same-H hidden-representation ablation

The layer-11 `q_proj`, rank-4 ablation reuses the existing adapter, raw per-sample A/B gradients,
fixed 2,000/1,000 split, and target-validation mean gradient. First cache the complete prompt-only
input sequence of the target module:

```bash
.venv/bin/python evaluation/cache_prompt_hidden_sequences.py \
  --experiment predictor/single_layer/results/layer11_qproj_rank4_direct/formal_2000_1000 \
  --expected-module-suffix layers.11.self_attn.q_proj \
  --device cuda:0
```

Then run the ablation:

```bash
.venv/bin/python evaluation/evaluate_hidden_representation_ablation.py \
  --experiment predictor/single_layer/results/layer11_qproj_rank4_direct/formal_2000_1000 \
  --device cuda:0
```

All representations are derived from the same cached target-module-input tensor `H`: `H[-1]`,
`H.mean(0)`, or a learned second-order core. The primary second-order representation is
`((H @ A2.T).T @ (H @ B1)) / T`; the unnormalized sum is retained only as a length-confounded
control. Primary pooled predictors use width 256. Width-318 last/mean predictors are additional
parameter-count controls for the second-order model. Every representation is trained with joint,
A-only, and B-only factor-wise cosine objectives on the same fixed 1,800/200 internal split; the
1,000 candidate examples are test-only. Outputs are written under
`formal_2000_1000/hidden_representation_ablation/`.

## Setup


```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The YAML configs contain a local model snapshot path. Change `model.path` and
`tokenizer.path` to your own Qwen2.5-1.5B-Instruct checkout before running.
GPU execution is expected for gradient extraction, SFT, and vLLM evaluation.

## Minimal reproduction path

The commands below illustrate the main stages. Output directories must be new;
the scripts intentionally avoid overwriting completed experiments.

```bash
# 1. Warm up a single-layer LoRA adapter.
python training/warmup_lora.py \
  --config configs/gradient_geometry/qwen2.5_1.5b_layer5_oproj_rank4_direct.yaml \
  --output runs/layer5_oproj/warmup \
  --device cuda:0

# 2. Extract matched prompt hidden states and per-example raw gradients.
python evaluation/extract_single_layer_raw_gradients.py \
  --config configs/gradient_geometry/qwen2.5_1.5b_layer5_oproj_rank4_direct.yaml \
  --adapter runs/layer5_oproj/warmup/adapter \
  --output runs/layer5_oproj/formal \
  --device cuda:0

# 3. Fit and evaluate the train-only-CV Ridge predictor.
python evaluation/evaluate_direct_raw_gradient_ridge.py \
  --experiment runs/layer5_oproj/formal \
  --device cuda:0

# 4. Measure candidate ranking against a held-out target gradient.
python evaluation/evaluate_target_alignment_spearman.py \
  --config configs/gradient_geometry/qwen2.5_1.5b_layer5_oproj_rank4_direct.yaml \
  --adapter runs/layer5_oproj/warmup/adapter \
  --experiment runs/layer5_oproj/formal \
  --device cuda:0
```

For the full layer/module/rank sweep, use
`evaluation/run_qvo_layer_rank_sweep.py`. SFT is implemented in
`training/train_lora_sft.py`; its selected and random branches must use the
same hyperparameters.
