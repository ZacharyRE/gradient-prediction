# Results

This document is the compact, maintained record of the important completed
experiments. Metrics are held-out unless stated otherwise.

## 1. Single-layer gradient prediction

### Layer/module/rank sweep

The sweep uses 2,000 predictor-training examples, 1,000 candidate-test
examples, and disjoint LoRA warmup data. Ridge hyperparameters are selected by
train-only five-fold cross-validation.

| Configuration | Mean-gradient baseline | Ridge cosine | Gain | Target Spearman |
|---|---:|---:|---:|---:|
| layer 5, `o_proj`, rank 4 | 0.6613 | **0.7203** | **+0.0589** | **0.5956** |
| layer 5, `q_proj`, rank 4 | 0.5094 | 0.5576 | +0.0481 | 0.5546 |
| layer 11, `o_proj`, rank 2 | 0.7207 | **0.7456** | +0.0249 | 0.4424 |
| layer 11, `q_proj`, rank 4 | 0.5651 | 0.6051 | +0.0400 | 0.4796 |
| layer 17, `o_proj`, rank 4 | 0.4934 | 0.5144 | +0.0209 | 0.3574 |
| layer 27, `q_proj`, rank 4 | 0.1213 | 0.1322 | +0.0109 | 0.2625 |

Across the complete Q/V/O sweep, Ridge improves over the mean-gradient
baseline in every configuration. Earlier layers generally provide stronger
target-ranking signal. The best reconstruction configuration is not the best
ranking configuration, so target Spearman is the primary selector metric.

The full 27-row table is generated locally at
`predictor/single_layer/results/qvo_layer_rank_summary.md`.

### LoRA factor asymmetry

The A factor is consistently easier to predict than B. In many settings,
A-factor cosine is 0.8–0.97 while B is substantially lower. Consequently, the
full A+B cosine partly reflects an easy shared A direction; B and mean-centered
residual metrics are more diagnostic of sample-specific information.

### Predictor and representation ablations

| Ablation | Result | Interpretation |
|---|---|---|
| Ridge vs MLP | layer-17 `o_proj`: 0.5144 vs 0.5087 cosine | Added nonlinearity did not help |
| Separate A/B Ridge penalties | 0.5153 cosine, rho 0.3723 | Only a small improvement |
| Rank-64 linear bottleneck | 0.5156 vs full Ridge 0.5167 | Predictable variation is strongly low-dimensional |
| Predictor data 2k→5k | 0.5167→0.5242; rho 0.3180→0.3420 | More data helps modestly |
| Mean vs last-token pooling | Mean pooling usually lowers cosine and rho | Last token is the stronger default |
| Learned second-order pooling | Below matched last-token predictor | More complex sequence pooling did not help |
| Exact module input vs post-layer state | Cosine changes are usually within ±0.005 | Hidden source matters less than layer/module choice |

For layer-5 Q/V, post-layer hidden states improve ranking rho by roughly
0.04–0.05. The strongest point remains layer-5 rank-4 `o_proj` with post-layer
hidden state.

## 2. Data selection

### MATH-only

- Candidate pool: 4,172 MATH-train examples.
- Target anchor: 200 disjoint MATH examples.
- Selected/random size: 2,000 each.
- Selected/random overlap: 959.
- Mean predicted score: selected 0.9608, random 0.8636.
- Leakage checks: passed.

The selected set is not distribution matched to random. It is shorter on
average (262.8 vs 354.5 tokens) and over-represents Algebra while
under-representing Geometry and Intermediate Algebra. Pure top-k selection is
therefore confounded by length, topic, difficulty, and redundancy.

### Cross-domain audit and mixed selection

A frozen MATH-trained predictor evaluated on GSM8K has:

- raw-gradient cosine 0.3351, below the MATH mean-gradient baseline of 0.3556;
- negative R²;
- target-ranking Spearman 0.2836.

There is weak cross-domain ranking signal, but no reliable cross-domain
gradient reconstruction. Mixed selection therefore uses fixed source quotas
(1,000 MATH + 1,000 GSM8K), but a domain-specific predictor remains preferable.

## 3. SFT

Selected and random branches always share hyperparameters; only their training
examples differ.

### MATH-only, Q/K/V/O LoRA

Configuration: all transformer layers' Q/K/V/O projections, rank 16, alpha 16,
dropout 0.05, learning rate 2e-5, one epoch, effective batch size 16, 125
optimizer steps, BF16.

| Evaluation | Base | Selected | Random | Selected − random | Paired p |
|---|---:|---:|---:|---:|---:|
| MATH common-unseen 1k | 48.3% | **35.5%** | 32.8% | **+2.7 pp** | **0.0055** |
| MATH-500 | 56.2% | 36.4% | **37.8%** | −1.4 pp | 0.3817 |
| GSM8K | 74.37% | 60.73% | **61.94%** | −1.21 pp | 0.1523 |

The common-unseen result is the clearest evidence that gradient selection is
not random noise. However, both SFT branches degrade substantially from the
base model, and selection does not improve either public benchmark.

### Mixed MATH + GSM8K, all-linear LoRA

Configuration: all transformer linear modules, rank 32, alpha 64, dropout
0.05, learning rate 2e-5, one epoch.

| Evaluation | Base | Selected | Random | Selected − random |
|---|---:|---:|---:|---:|
| MATH-500 | 56.4% | **30.2%** | 30.0% | +0.2 pp |
| GSM8K | 74.53% | 60.12% | **61.87%** | −1.74 pp |

An earlier learning-rate-2e-4, three-epoch run overfits after epoch 1. Its best
selected-over-random point is MATH-500 at epoch 2 (+3.4 pp, p=0.107), but the
gain is neither significant nor persistent.

## 4. Conclusions supported by the data

1. Prompt hidden states contain a useful, low-dimensional signal about the
   direction of a same-layer LoRA gradient.
2. Gradient reconstruction and candidate ranking are different objectives;
   reconstruction cosine should not be the sole model-selection metric.
3. A single-layer alignment proxy has not yet been validated as utility for an
   all-layer SFT update.
4. Independent top-k alignment creates distribution shift and redundant
   selections.
5. The current SFT degradation is larger than the selected-vs-random effect,
   so the SFT control must be stabilized before claiming end-to-end gains.
