# What each method does

**Oracle32, Oracle16 and CV16 describe true-gradient examples per update, not the number of training steps.** For example, `CV16 @256` means 256 adaptation updates, each predicting all 32 examples and correcting with 16 true gradients. All adaptation horizons are additional to the common 32-step warmup.

`State256` names the state-aware predictor's hidden width of 256. Its fitting set contains 512 unique examples evaluated at eight development teacher states (4,096 example-state pairs). The inherited p123 predictor instead used all 2,050 original fitting examples at one teacher state. `State256 CV16 first 32 @256` uses the frozen eight-state predictor with 16 true correction examples for 32 updates, then runs 224 Oracle16 updates. The later 12-state/2,048-input development ablation is a separate predictor and does not replace this frozen method.

Every adaptation recipe starts from the shared warmup adapter weights with a **fresh AdamW optimizer state**; warmup moments are not restored. The early-cutoff recipe preserves its adaptation optimizer state when switching from CV16 to Oracle16 at update 32.

MNLI uses the model's chat template with a task-specific instruction: A = entailment, B = neutral, C = contradiction, and only the option letter inside `\boxed{}` is requested. Full premise/hypothesis inputs are retained. Generated-answer accuracy and teacher-forced ABC probabilities are evaluated separately.

The selected gradient networks are two-layer Transformer encoders with eight attention heads and attention over the full valid sequence. The inherited p123 model has width 512 and takes current output-projection activations. State256 has width 256 and takes the current post-attention residual hidden state, the explicit current LoRA output contribution, and 8-dimensional target-token conditioning. Both predict 1,536-dimensional activation gradients, which are converted to LoRA parameter gradients by the exact local chain rule. The selected anchored behavior model is a separate MLP with two 128-wide hidden layers and three ABC outputs.

Parameter counts are **5,787,136** for p123, **2,245,888** for State256 and **1,197,056** for the anchored behavior model, excluding buffers. All three passed strict CPU checkpoint loading. [Exact checkpoint configurations, counts and hashes](results/selected_predictor_architectures.json).

| Method | Update | Extra preparation |
|---|---|---|
| Warmup baseline | Stop after the common 32 warmup updates | None beyond warmup |
| Oracle32 | Exact gradient of each 32-example batch | Shared implementation caches only |
| Oracle4/8/16 | Uniform subset of 4/8/16 examples, rescaled to estimate the 32-example batch gradient | No learned predictor |
| CV4/8/16 | Predict all 32; add a rescaled true-minus-predicted correction on 4/8/16 uniformly sampled examples | Frozen predictor |
| Sparse calibration | CV with delayed predictor updates from correction labels already paid for | Calibration compute is timed; no extra teacher labels in this arm |
| Explicit-state predictor | Supply the current LoRA output contribution explicitly alongside the current post-attention residual hidden state | Fit on 8 development teacher states |
| Extended state coverage | Same state256 architecture with four additional high-learning-rate Oracle states through 512 updates | Original 8 low-rate states plus 4 additional states;40 versus original 60 epochs gives exactly the same 15,840 optimizer steps, checked from receipts |
| Mean-aware fitting/selection | Same state256 architecture; include batch-mean error in checkpoint selection, with mean-loss weight0 versus 4 | Same teacher labels; two additional matched 30-epoch fits |
| Scalar/Sobolev critic | Learn a scalar suffix loss; differentiate it to produce synthetic gradients | Teacher losses/gradients and derivative-aware fitting |
| Causal token critic | Approximate suffix answer logits; obtain gradients through its loss | Distillation and/or gradient fitting |
| Copied suffix | Retain teacher width, layer 8 MLP, decoder blocks 18/27 and answer readout | Distillation, optionally gradient supervision |
| Factor predictor | Approximate `C = GᵀX` in a compact basis, then reconstruct gradients using current LoRA factors | Teacher gradients and basis fitting |
| Class covariance | Use only class-mean covariance; selected factor checkpoint had a zero learned residual head | Class-template construction; no neural residual inference |
| Factor scales | Predict a few bounded magnitudes for class-covariance factors | Training-only PCA and weighted ridge fitting |
| Position templates | Class/relative-position mean activation gradients combined with each example's exact input features | One-state or multi-state gradient labels; no optimizer fitting |
| Minibatch table | Reuse stale per-example gradients as predictions within the current 32-example batch | Optional paid full-dataset bootstrap |
| Full-data table | Maintain the prediction sum over the entire candidate dataset and correct with an IID subset | Paid bootstrap or zero initialization, separately tested |
| Local linearization | Cache two ABC logit-difference Jacobians; approximate logits under changed LoRA outputs | Jacobian construction and any refresh are timed |
| Periodic true updates | Alternate true-gradient updates with predicted, held, or zero-gradient controls | Any predictor/anchor preparation is explicit |
| Behavior readout | Predict forced-format ABC logits from input features and current adapter effects | Separate small-model fitting; this is not free-generation accuracy |
| Anchored behavior readout | Retain the initial agent's exact ABC logits and predict only a centered change; subtract the same network's zero-change output | One paid initial-teacher readout per target start/evaluation population, reused across candidate adapters, plus separate fitting |
| Early predictor cutoff | Use state256 CV16 for 32 updates, then Oracle16 with the same Adam state and sampling | Initial predictor preparation is still charged; diagnostic control for loss of late-step variance reduction |

For a minibatch with N=32 examples and a uniform correction subset S of size k, the raw CV gradient sum is:

`sum_i predicted_gradient_i + (N/k) × sum_{i∈S}(true_gradient_i − predicted_gradient_i)`.

It is divided by the minibatch's supervised-token count, then clipped and passed to AdamW. Conditional on the current predictor and minibatch, the correction is unbiased before numerical rounding. This does not guarantee downstream accuracy: variance, clipping, Adam, training duration and distribution shift all matter. Delayed calibration uses current labels only for subsequent updates.

The initial direct/state predictor checkpoints were selected by centered residual variance, which is appropriate to CV but does not constrain a constant mean-gradient bias. Pure and periodic prediction failures using those checkpoints do not establish that every standalone-surrogate objective must fail. A separate same-architecture control selects expected batch32 mean-gradient error, including bias, and compares an added mean-loss penalty against no penalty. Both use 30 epochs and only the existing development labels. The state256 package also changes label conditioning, representation and fitting data versus the inherited predictor; a gain cannot be attributed solely to the explicit-state feature without matched ablations.

Token and copied-suffix critics use the union of supervised answer tokens as their output vocabulary. Their pure cross-entropy updates therefore omit probability mass outside that union; this is another approximation besides suffix compression. CV corrections still use the original full-vocabulary teacher objective. Pure failures should not be attributed solely to network depth, and these critics are not presented as exact full-vocabulary replacements.

The full-data table uses the analogous prediction sum over the entire dataset and its total supervised-token denominator. Its IID controls use the same sampling rule. It is inspired by variance-reduced optimization; convex SAGA convergence guarantees are not claimed for this LoRA/Adam experiment.

# What is shared with Oracle

Only one layer 8 output projection has trainable LoRA weights. Therefore the preceding frozen computation can be cached exactly. Oracle and CV receive this same optimization. This cache would need reconsideration if earlier layers were trainable.

The optional CUDA Graph backend pads full sequences to64-token buckets and uses an eager fallback when a shape exceeds the memory policy. It never truncates real inputs. Graph and eager-static modes use the same padding and arithmetic; native execution uses its original padding and remains an independent quality/time control. Graph recipes run in separate CUDA processes after a multi-recipe graph-lifetime failure. Lazy capture, warmup, audits and bootstrap are charged where they occur.

# Two different surrogate endpoints

The gradient predictor is a surrogate for an update signal. CV still queries the real teacher for corrections, so successful CV is evidence for a hybrid training procedure, not automatically a standalone replacement for the agent. A separate behavior readout and actual candidate-selection test are required for that stronger interpretation. This study tests one supervised classification adaptation setting, not general multi-turn agent behavior.

1. **Direct training acceleration:** deploy the adapter produced by approximate updates; compare its generated-answer accuracy and elapsed time with matched and independently tuned true-gradient controls.
2. **Data selection:** approximate training on each of five fixed candidates, score each on selection_dev, freeze the choice, then measure the chosen candidate's actual Oracle outcome on selection_holdout. The deployment pipeline includes one final Oracle training run. Exhaustive Oracle selection can reuse its already-trained chosen adapter.

The frozen analysis additionally reports `direct_deploy_accuracy` for the selected simulator-produced adapter. Keeping that adapter avoids final Oracle retraining, but its accuracy can differ from the reference Oracle trained on the same chosen data. This secondary variant is reported for predictor and true-gradient controls alike. Its cost table uses candidate-training and readout component sums; the separately measured complete-workflow benchmark retains the registered final-Oracle endpoint.

The random-subset and genre-subset families are separate experiments. Rank correlation is reported alongside candidate spread and selection regret: low regret can be trivial when all candidates perform similarly. A cheap exact teacher forced-format readout is a final control for the learned behavior readout. The behavior model without explicit adapter-delta features was evaluated during development; final model selection retains only the expanded-coverage anchored128 model, alongside exact-current-teacher and unchanged-initial controls. No forced-format result is relabeled as free-generation accuracy or general agent behavior.

The anchored readout is exactly equal to its paid initial-teacher logits when the adapter has not changed. Its meaningful test is whether it predicts subsequent behavior changes better than retaining those initial logits unchanged. Initial classification accuracy alone cannot establish surrogate learning. Model selection compares the same normalized logit plus within-example state-change error on predictor_dev; a model trained without the change penalty is rescored with both terms before comparison. The initial teacher readout is charged once per start and readout dataset, not once per candidate, and is not included in offline fitting time.

After the original anchored pilot failed to track later/higher-LR states, the same 64/128-wide architectures were fitted with four additional Oracle states. All eight old/new behavior candidates were then rescored on the same 12 states,129 predictor-dev examples and training-derived normalizers. Their original 8-state and expanded12-state fit scores are not directly comparable. The common-score selection chooses anchored128 with expanded coverage; the unchanged-initial and exact-current-teacher controls remain mandatory.

# Cost boundaries

- **Preparation plus online updates:** frozen-prefix cache, predictor load where recorded, table/Jacobian bootstrap, graph capture, calibration and updates. Checkpoint diagnostics are separate.
- **Standalone replay:** fresh-process imports, model/data/predictor loading, preparation, updates and final save; final weights must exactly match the evaluated adapter. Common target warmup, inherited predictor construction and evaluation remain separate.
- **Predictor preparation:** historical label collection, fitting, prerequisite teacher trajectories and source warmups are inventoried. Shared dependencies are charged once. These receipts are not a clean cold-start benchmark or a theoretical minimum.
- **Selection:** report all candidate adaptation/readout costs and the final Oracle run, versus exhaustive Oracle search and cheap controls. Component-sum estimates are distinct from the registered complete-workflow benchmark: start 302, five random candidates, six fixed simulator/readout combinations, three fresh repetitions. It includes process transitions and loading, and requires exact frozen weights, readouts and choices. It does not establish end-to-end cost for other starts or genre candidates.
- **FLOPs:** count matrix/convolution arithmetic, including actual graph replays and capture warmups/audits. Capture tracing itself does not execute those kernels. Nonlinearities, reductions, normalization, optimizer scalar operations and offline FLOPs not explicitly profiled are not silently inferred from elapsed time.
- **Research overhead:** method search, unsuccessful runs, repeated evaluation and construction of ground-truth Oracle outcomes are separate from a hypothetical deployment pipeline, but remain part of this study's resource use.
