# Experiment map and research plan

## Research hypothesis

For a candidate example \(x_i\), use a prompt representation \(h_i\) to predict
its single-layer LoRA gradient \(\hat g_i=f(h_i)\). Score the candidate against
a held-out target direction \(\bar g_T\):

\[
s_i = \cos(\hat g_i, \bar g_T).
\]

The operational hypothesis is stronger than gradient prediction itself: data
with high \(s_i\) should produce more useful downstream updates than random
data under a fixed training budget.

## Completed attempts

### Gradient targets

- Full flattened raw LoRA A+B gradient; no sketch or SVD in the primary runs.
- Rank 2 and rank 4.
- Q, V, and O projections across layers 5, 11, 23, and 27.
- Layer-17 Q/V/O and `down_proj` reference experiments.

### Predictors

- Ridge with train-only cross-validation.
- Separate A/B Ridge penalties.
- One-hidden-layer MLP.
- Shared nonlinear A/B bottleneck.
- Shared and factor-specific reduced-rank linear Ridge bottlenecks.
- Predictor-training scale-up from 2,000 to 5,000 examples.

### Prompt representations

- Last prompt token.
- Mean pooling.
- Exact target-module input vs same-block post-layer state.
- Learned second-order mean and sum representations.

### Selection and downstream training

- MATH-only top-k predicted target alignment.
- Equal-quota MATH/GSM8K selection.
- Frozen-predictor cross-domain audit.
- Selected vs random SFT with all-layer Q/K/V/O LoRA.
- Selected vs random SFT with all-linear LoRA.
- Base, selected, and random evaluation on MATH-500, GSM8K, and a common
  unseen MATH-train holdout.

## Main unresolved causal chain

```text
hidden state predicts local gradient
                ↓ established
predicted gradient ranks true alignment
                ↓ established in-domain
local alignment predicts all-layer update utility
                ↓ unverified
selected SFT improves generalization
                ↓ not observed
benchmark accuracy exceeds random and base
```

## Prioritized next experiments

### 1. Stabilize the SFT control

Tune only on random MATH data first. Test learning rates 2e-6, 5e-6, and 1e-5
with checkpoints at 25, 50, and 125 optimizer steps. Prefer Q/K/V/O rank 8 or
16 before all-linear rank 32. The acceptance criterion is no large MATH-500 or
GSM8K regression relative to base.

### 2. Match the selector and training parameter spaces

Run a diagnostic ladder:

1. selection and SFT both use layer-5 `o_proj`, rank 4;
2. selection uses layer-5 `o_proj`, SFT uses all layers' `o_proj`;
3. SFT expands to all layers' Q/K/V/O;
4. SFT expands to all-linear.

This isolates whether the signal transfers across layers and modules.

### 3. Add a true-gradient oracle

On a smaller pool, compare matched-size sets selected by:

- random sampling;
- length/type-matched random sampling;
- hidden-state similarity;
- predicted-gradient alignment;
- true-gradient alignment.

If the oracle fails, the alignment objective is the problem. If the oracle
works but the predictor fails, predictor ranking is the bottleneck. If both
work but all SFT branches degrade, the training recipe is the bottleneck.

### 4. Replace independent top-k with diversity-aware selection

Start with type/level/length quotas and clustering. Then test greedy residual
gradient matching, selecting examples that explain the remaining target
direction rather than repeatedly choosing near-duplicate aligned gradients.

### 5. Optimize for ranking or utility directly

Instead of reconstructing 12,288 gradient coordinates, predict the scalar
alignment score or train a pairwise ranker. Later, replace raw cosine with a
closer approximation to training influence, such as a validation-gradient dot
product or a preconditioned influence score.

## Reporting checklist

Every selection experiment should record:

- exact candidate, anchor, train, dev, and test partitions;
- leakage checks and selected/random overlap;
- source, type, level, token-length, and loss distributions;
- true-gradient oracle and matched-random baselines where feasible;
- at least three SFT seeds for final comparisons;
- paired confidence intervals or exact tests;
- performance relative to both random SFT and the untouched base model.
