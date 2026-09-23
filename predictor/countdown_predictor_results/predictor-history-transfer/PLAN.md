# Predictor history transfer: frozen pilot protocol

Question: at an identical main-model state and on identical unseen examples,
does retaining predictor weights from earlier calibration episodes improve
gradient prediction and the usefulness of a subsequent model update?

## Fixed comparison

- Main model: Qwen2.5-0.5B-Instruct, frozen base, layer-8 o_proj LoRA,
  rank=alpha=64. Use existing real-gradient oracle trajectories, not models
  updated by either predictor condition. Pair oracle seeds 101/102/103 with
  pretrained Y+mask+position predictor seeds 123/124/125.
- Common initial predictor: existing one-step pretrained checkpoint (8192
  training examples; 256 development examples). It reads complete
  teacher-forced question+reference answer hidden states, not prompt only.
- History episodes occur at oracle steps 0,1,2,4,8,16. Step 0 denotes the
  common teacher32 warmup state. Each episode has four new calibration
  examples and sixteen new selection examples; no repeats across episodes.
- Carry retains the selected predictor weights after each history episode.
  Reset restarts from the same pretrained predictor before each episode.
  Both undergo all history episodes with identical data and optimizer-step
  budgets. Predictor AdamW state resets at every episode in both conditions.
- At oracle steps 16 and 32, evaluate four independent probe episodes. Each
  starts from the history available strictly before that state. Probe
  calibration never changes the ongoing history. Each probe has four
  calibration examples, sixteen selection examples, and 28 held-out
  examples. The update batch is the four calibration plus 28 held-out items.
- Step 32 is primary; step 16 is secondary. All four probes are reported,
  with paired seed-level summaries. Shared examples across seeds are not
  counted as independent new examples.

## Calibration and update

- Calibration: exactly 30 optimizer steps on four examples, AdamW lr=1e-4,
  weight decay=.01, clip=1; do not stop early. Select among epochs 0..30 by
  the existing independent-validation score: factor-balanced per-example
  A/B relative squared error plus aggregate A/B relative squared error.
- Training objective: factor-balanced A/B relative MSE + .25 activation G
  MSE normalized by the original pretrained target scale. Main model and
  LoRA weights remain fixed during calibration.
- Measure before/after calibration. Held-out 28-example gradients cannot
  select a checkpoint or otherwise affect calibration.
- For each probe, apply exactly one real model update from the SAME original
  LoRA state using carry, reset, unchanged pretrained predictor (frozen), or
  full true gradient (oracle). The predicted update uses all 32 examples;
  current calibration labels are only four training plus sixteen selection.
- Each intervention starts a fresh matched AdamW optimizer: lr=3e-4,
  weight decay=0, clip=1, no scheduler. Report this fresh-optimizer scope;
  it does not reproduce continued optimizer dynamics in a long rollout.
- Reconstruct A/B predicted gradients analytically with current X,A,B and
  divide by the batch's total supervised-token count. Audit the true
  reconstruction against actual autograd before applying interventions.

## Data and endpoints

- Draw new problems from the local original Countdown source, excluding the
  union of all named prior project data partitions by sorted number multiset,
  including all predictor train/extra/dev and earlier evaluation sets.
- Use exact-solver reference solutions. Balance three- and four-number
  problems within every split, including calibration4 and held-out28.
  Fix selection seed 202609231; write source indices and file hashes.
- Fresh downstream loss set: 256 examples, shared across probes and seeds.
  Report full completion token CE and boxed-expression token CE before and
  after each intervention. This set cannot select models or hyperparameters.
- Fresh accuracy set: 1024 examples. Greedy generation, complete input,
  2048 NEW OUTPUT tokens maximum, exact arithmetic/inventory verifier.
- To bound generation cost, accuracy is prespecified for probe 0 at primary
  oracle step 32 only: 3 seeds x (baseline + 4 intervention methods)=15
  evaluations. This is a secondary sparse check, not a replacement for the
  four-probe loss and gradient analysis. Save every full prompt and answer.
- Primary descriptive endpoints: carry-minus-reset held-out gradient
  relative L2, and paired held-out completion-CE improvement after update.
  Cosine, norm, expression CE, before-calibration metrics and accuracy are
  supporting endpoints. Report seed/probe heterogeneity, not only a mean.
- If reporting a question bootstrap for accuracy, condition it on the three
  fixed trajectories and explicitly exclude seed uncertainty from the claim.

## Costs, limits and execution

Record true-gradient training/selection/diagnostic examples separately,
predictor optimizer steps, selection evaluations and synchronized GPU times.
Common initial pretraining and oracle trajectory construction are reused
sunk costs, not zero costs. A reset baseline that skips all history would
have a cheaper lifetime budget; report that distinction. Diagnostic true
gradients do not count as free deployment supervision. Do not infer overall
compute savings from this pilot.

This tests history transfer under fixed reference model trajectories. It does
not test self-generated trajectories, meta-learning objectives, multi-layer
scaling, cross-task transfer, or recursive self-improvement. Positive carry
results establish useful retained experience, not automatically a better
learning algorithm. Negative results do not invalidate all predictors.

Run preparation, then three independent seed jobs on available GPUs 5/6/7.
Audit matched-state, matched-data and no-truncation invariants before the
15 prespecified accuracy evaluations. Finish with a Chinese REPORT.md,
machine-readable paired results and standalone figures. No outcome-based
hyperparameter sweep is planned.
