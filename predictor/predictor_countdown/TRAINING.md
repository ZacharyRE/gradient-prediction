# Training configuration

## Model and objective

Base: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`. Freeze base parameters. Use completion-only next-token cross-entropy with complete solver solutions; normalize LoRA update gradients by the batch's total supervised-token count. Preserve all input tokens.

Predictor experiment: zero-based block 8, `self_attn.o_proj` only; LoRA A is 64×896, B is 896×64; rank=alpha=64, dropout=0. Base/LoRA parameters are FP32 with BF16 autocast during gradient collection. Collect the activation target G = d(sum CE)/dY. With scaling s=1:

```text
predicted dA = (G_hat B)^T X / N
predicted dB = G_hat^T (X A^T) / N
```

All real token positions participate, including prompt positions: prompt activations influence later supervised predictions. Evaluation of gradient fidelity is per question; actual update aggregation is token-weighted.

## Data roles

Number-multiset grouping separates training and evaluation questions, including different targets with the same numbers.

| Split | Questions | Role |
|---|---:|---|
| warmup | 512 | Real-gradient LoRA trajectory; step 32 is the shared start |
| predictor_train | 1,536 | Offline gradient supervision and preconditioner queries |
| predictor_extra | 6,656 | Expands the predictor pool to 8,192 questions |
| predictor_dev | 256 | Predictor and offline preconditioner selection |
| gradient_test | 256 | Frozen predictor evaluation |
| update | 1,536 | Subsequent LoRA updates using reference solutions |
| accuracy_dev256 | 256 | Update hyperparameter/checkpoint selection |
| accuracy test | 2,048 | Final question-only inference |

Multiple states repeat questions and increase oracle gradient calls, not independent question count. `sequence_multi` config's `n_train=3072` means 1,536 questions at each of two states. Feedback large combines 8,192 state-32 examples with 1,536 state-96 examples.

## Local MLPs

Both have width 512. For each token, independently project X and Z through LayerNorm → Linear(896,256) → GELU. Concatenate the resulting 512 values with a supervision indicator, relative position p, and log(1+100p)/5.

- `local32`: MLP 515→512→512→896, GELU between layers; 1,449,344 trainable parameters.
- `conditional32`: additionally concatenate two 128-dimensional projections of frozen embeddings (correct next token and mean supervised solution embedding); MLP input is 771; 1,695,232 parameters. Non-supervised positions have no correct-label embedding contribution before the projection.
- Multiply the final prediction by the fixed training-target RMS scale, approximately 0.0397123.

Train at state 32 on all 319,668 tokens from 1,536 questions. AdamW, LR=3e-4, weight decay=0.01, gradient clipping=1, seed=123. First 24 epochs: shuffled token batches of 4,096; activation-gradient MSE divided by target RMS squared. Next 4 epochs: batches of 8 complete questions; relative squared error on concatenated A/B gradients plus 0.1 times the normalized activation loss. Select minimum development per-question A/B relative MSE across checkpoints.

## Local sequence predictors

Use the label-conditioned features above, input projection to model width, bidirectional Transformer attention, then LayerNorm and linear output. The 512×2 architecture has 8 heads and 5,639,168 parameters. Unlike the MLP, it exchanges information across token positions. Its loss balances A and B separately: full parameter-gradient relative MSE plus activation MSE with weight 0.25.

| Run | Training coverage | Epochs | LR | Batch |
|---|---|---:|---:|---:|
| `sequence32_wide` | 1,536 questions, state 32 | 160 | 3e-4 | 16 |
| `sequence_large32` | 8,192 questions, state 32 | 100 | 1e-4 | 16 |
| `sequence_multi` | 1,536 questions at states 32 and 96 | 120 | 3e-4 | 16 |

`sequence_large32` continues from the selected `sequence32_wide` checkpoint: both stages are required to describe its training. The larger capacity control is width 768 with 4 layers (21,195,520 parameters). Exact recorded arguments are in configs.json.

## Feedback predictor

`feedback_large`: width 512, depth 4, 8 heads, 7,808,261 parameters; 80 epochs, batch 16, AdamW LR=2e-4, weight decay=0.01. Training covers states 32 and 96. It learns activation-dependent error transport using full-forward features and analytically computed readout/RMSNorm error; transposed causal attention conveys later-token error to earlier positions. Replay adds predictor-generated training states with fresh real-gradient supervision; it does not train on the separate update set.

## LoRA warmup and deployment updates

Warmup: AdamW LR=5e-4, weight decay=0.01, batch 32, microbatch 8, clipping=1, seed=101. The archived teacher run saves states 0/32/96; deployment starts at 32.

| Selected update | Optimizer | LR | Weight decay | Steps |
|---|---|---:|---:|---:|
| Local `sequence_multi` and short mean/true controls | SGD | 0.01 | 0 | 16 |
| Feedback large and long mean/zero/true controls | SGD | 0.01 | 0.01 | 384 |
| Real-gradient reference | AdamW | 5e-4 | 0.01 | 192 |

All use batch 32 and clipping=1. Predictor weights stay frozen during these updates. Recorded run configs may list longer training horizons; the table gives development-selected checkpoints. `Local-SGD16` uses `sequence_multi`, not the fixed-state `sequence_large32` headline gradient model.

The additional learned preconditioner uses delta = −eta [diag(softplus(d)) + UU^T] g, rank(U)=4, eta=0.1 and radius=2. It has 573,440 parameters; fit for 1,000 steps at LR=0.001 using post-update query CE + 0.005||delta||². Context/query are separated within each episode, drawn from predictor_train; select on predictor_dev. Deployment generates one update from the independent update set. Its matched fixed-mean baseline receives the same-sized learned updater.

## Earlier single-block SFT configuration

Separate feasibility study: 4,096 train / 1,024 development / 2,048 test, solver solutions, seeds 101/102/103. All projections means q/k/v/o and gate/up/down. Rank=alpha=64, dropout=0; batch 32, microbatch 8; AdamW weight decay=0.01, clipping=1; 3% warmup then cosine decay over 4 epochs / 512 updates; FP32 parameters with BF16 autocast. Development selected epoch 4 for both main configurations: all 24 blocks at LR=2e-4; block 8 only at LR=5e-4. These settings and budgets differ from the later single-module predictor experiment.
