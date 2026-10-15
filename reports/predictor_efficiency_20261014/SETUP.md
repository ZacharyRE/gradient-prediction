# Predictor efficiency study: setup and interpretation

This study tests whether a LoRA-gradient surrogate can retain downstream accuracy while reducing measured training time, including after 32 updates. It covers **Qwen2.5-1.5B-Instruct on PAWS and MNLI**. It adds no new 0.5B or 7B results.

The surrogate approximates the training update. We measure the resulting agent's capability by applying those updates and evaluating the actual model.

All reported update-step counts start **after** the shared 32-step warmup. For example, a 256-update trajectory performs 256 additional steps from that fixed adapter.

| Item | Setting |
|---|---|
| Hardware | One H200; cooperative reservation and recorded occupancy |
| Trainable weights | Layer 8 attention output projection; LoRA rank/alpha 64; 196,608 parameters |
| Precision | FP32 parameters; BF16 autocast |
| Starting point | Fixed task-specific 32-step warmup, batch 32, seed 201, learning rate 0.001 |
| Update batch | 32 examples; AdamW, weight decay 0, gradient clipping 1 |
| Paired update seeds | 101, 102, 103 for replicated comparisons |
| Paired original predictor seeds | 123, 124, 125 |
| Development selection | Generated accuracy on 512 PAWS / 513 MNLI examples |
| Final readouts | PAWS test: 8,000 examples; MNLI validation_matched: 9,815 examples (called `test` locally); recipes frozen before confirmation/test; source groups remain disjoint |
| Evaluation | Greedy generation, full input token sequences, at most 256 **new output tokens** |

| Data split (examples) | PAWS | MNLI | Purpose |
|---|---:|---:|---|
| Warmup | 260 | 258 | Produce the fixed starting adapter |
| Update | 4,096 | 4,097 | Training trajectories; reshuffled passes |
| Predictor train | 2,055 | 2,050 | Offline gradient-surrogate fitting |
| Predictor development | 130 | 129 | Select the fitted predictor using gradient error |
| Accuracy development | 512 | 513 | Select update recipes and stopping points |
| Confirmation | 512 | 512 | Frozen-recipe readout |
| Gradient test | 132 | 129 | Diagnostic only; neural trajectory audits use its first 16 examples |
| Final evaluation | 8,000 | 9,815 | Frozen-recipe readout |

All split pairs have zero source-group overlap. Final evaluation contains 1,616 PAWS / 3,334 MNLI source groups; uncertainty calculations preserve these groups. A few specialized fits use an explicitly recorded training subset, such as 512 predictor-training examples per state in the multi-state experiment.

The frozen runs produce 25,600 confirmation and 445,375 final-test generations across the 50 adapters. No output reaches the generation cap, and strict-format accuracy equals the reported accuracy for every adapter. Generation-time input-token verification is recorded in the published evaluation protocols.

**Oracle32** computes the true gradient on all 32 examples. **Oracle4/8/16** estimates the same batch gradient from a uniformly sampled subset. **CV4/8/16** combines predictions for all 32 examples with true-minus-predicted corrections on 4/8/16 sampled examples. The correction coefficient and predictor are fixed before drawing the current subset; calibration uses training labels and affects later updates. This gives an unbiased raw-gradient estimator under its sampling assumptions, not an unbiased Adam/clipped update.

The reused neural predictor is a two-layer, width-512, eight-head Transformer. It receives the trainable projection's output activations plus position/mask features and predicts the backpropagated activation gradient. The projection input and current LoRA factors convert that prediction into LoRA-A/B gradients. PAWS uses the inherited per-example-normalized fit; MNLI uses the inherited `globalfit` checkpoint, which normalizes factor losses using training-set statistics. New refits explicitly vary the input representation, width, labels, and loss; exact settings are in the recipe inventory.

The class-only online alternative uses a two-/three-dimensional one-hot task-label vector to predict the flattened LoRA gradient with recursive least squares. It learns from previous correction samples, with forgetting factor 0.99 in the replicated original version. It has 393,216 / 589,824 predictor coefficients on PAWS / MNLI and needs no offline predictor fit. Its fitting and the work required to retain correction labels are included in online time. It is a small gradient surrogate, not a replacement agent that directly generates answers.

Both predictor methods and true-gradient baselines use the optimized supervised-token language-model head and appropriate microbatching. Full transformer inputs are retained. FP32 agreement with the original loss implementation is within about 3e-6 relative gradient error; BF16 execution paths differ by about 1%, so all comparisons are evaluated again under the optimized backend.

Recipe `token_budget` values control microbatch grouping; an individual sequence remains intact. They are not prompt truncation limits.

We compare independently tuned learning rates and stopping points, as well as matched configurations. A predictor beating a slow Oracle implementation or a poorly chosen Oracle stopping point is insufficient evidence of useful acceleration. Oracle4/8/16 and no-predictor gradient reuse/smoothing controls isolate the predictor's contribution.

## Cost definitions

- **Online updates:** synchronized update time, including predictor inference, true-gradient correction, and online calibration. Intermediate evaluation and gradient diagnostics are excluded.
- **First-use adaptation:** online updates plus required prefix/body cache construction. Standalone timing replays also record loading and final serialization separately from the update sum.
- **Cold predictor preparation:** teacher-label collection and predictor fitting; reported separately and included when claiming first-ever-use savings. Multi-state fitting additionally requires a teacher training trajectory.
- **Research overhead:** screening, repeated evaluations, audits, and failed attempts; not disguised as deployment cost.
- **FLOPs:** counted matrix arithmetic, including supported fused operators. Scalar, nonlinear, normalization, optimizer arithmetic and the small RLS linear solves are outside this counter. Each receipt lists operator coverage and uncounted operators. Instrumented replay times are not benchmark times.

Standalone timings use local model artifacts and an already running host; operating-system file caches are not flushed. They include the script's imports/loading and final serialization, while online timings describe adaptation with the base model loaded.

The registered three timing rounds use one replay per paired update/predictor seed and rotate method order between rounds. Reported ranges include both data-order effects and execution variability; they are not three repetitions of one identical sample order. Replayed adapter weights must match the corresponding evaluated checkpoint before its accuracy and time are combined.

Eight matching FLOP profiles were registered with the frozen plan. Two supplementary profiles—MNLI CV16 and early-stopped Oracle32, both at 256 updates—were added after readouts to complete the cost comparison of those already-frozen recipes. This adds no accuracy selection or configuration changes. The separate [cost registration](provenance/supplementary_cost_registration.json) records their scope and commands.

The cached variants store features for the fixed update pool on this H200. Their preparation cost is included each time a fresh standalone trajectory is timed. New examples require new features, and the frozen-prefix shortcut relies on this single trainable projection; the measured speed does not establish the same benefit for all-layer LoRA or a stream of new agent observations.

The reused original predictor's historical label-collection plus fitting time was **84.18 seconds for PAWS** and **108.44 seconds for MNLI** for predictor seed 123, excluding separate held-out gradient diagnostics. These are recorded components, not a separately timed end-to-end cold pipeline or lower bounds for a newly optimized implementation. Reuse can amortize preparation, but preparation is not free.

The complete preparation ledger is `results/predictor_preparation_costs.json`. A few initialized linear-predictor screens reuse a subset of old gradient labels; that subset's collection time was not separately measured and is marked as such in the inventory. Their short initialization timer is not a complete cold-start cost.

Any amortized saving assumes repeated compatible adaptations can reuse the same fitted predictor. This study does not demonstrate reuse across different tasks, model sizes, or arbitrary starting adapters. Repeating experimental seeds is not itself evidence of such deployment reuse.

Multi-state fitting uses an already-computed Oracle teacher trajectory on the same task. That trajectory itself provides an adapted model, so its cost and resulting adapter are essential controls for first-use training. Faster subsequent surrogate updates would motivate transfer experiments; they would not by themselves demonstrate a cheaper first solution to the original task.

## Limits on conclusions

The evidence is specific to one trainable LoRA projection and these two classification tasks. Transfer to all-layer adapters, other model sizes, or agent interaction trajectories requires separate experiments.

The practical target is a downstream gap within one percentage point of independently tuned Oracle32. Point agreement alone does not establish noninferiority. Final comparisons use paired update-seed and whole-source-group bootstrap intervals, conditional on one fixed warmup. These test populations were inspected in earlier studies, so the results remain exploratory.

GPU occupancy is sampled throughout. A foreign process established a small CUDA context during the study despite the cooperative reservation. Accuracy exploration may continue under a recorded, identity-checked exception; overlapping timings are flagged and require a clean replay for final speed claims. This reservation is not administrator-enforced exclusive mode.
