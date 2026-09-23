# Outcome-informed learning-rate diagnostic

Added after inspecting the completed main-protocol gradient/loss records,
while the prespecified accuracy runs were still in progress. At primary
state32 the true-gradient one-step control worsened mean held-out completion
CE (mean improvement -0.002956); the main LR=3e-4 may obscure useful updates.

Repeat the SAME 24 interventions at LR=1e-4, changing no data, predictor
history, calibration objective, selection rule, or model state. Reconstruct
the selected predictors deterministically from saved history and assert that
their selected validation scores/epochs agree with the main run. Check the
probe-0 predictor tensors against the saved main-run predictor tensors.

Report all four methods and both states on the existing loss256 set. The
original 15 accuracy evaluations remain at LR=3e-4; no additional accuracy
generation or model selection is authorized by this control protocol.
This is an exploratory diagnostic on a reused loss set, not independent
confirmation and not a prespecified hyperparameter comparison.

The original PLAN.md and main-protocol results remain unchanged.
