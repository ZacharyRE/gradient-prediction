# Research Roadmap

[Project overview](../README.md) · [Results index](RESULTS.md) · [Predictor map](../predictor/README.md)

## Current question

Can a learned gradient predictor remain useful for LoRA optimization as model parameters change, and can that approach extend from one module to all layers?

## Research stages

| Stage | Evidence so far | Main reference |
|---|---|---|
| Static gradient predictability | Hidden states contain gradient-direction and ranking information; data-selection transfer remains limited | [Earlier MATH/GSM8K study](DATA_SELECTION_RESULTS.md) |
| Localized learning and state changes | Localized SFT is feasible; fixed-state prediction quality can deteriorate at later LoRA states | [Early Countdown study](../predictor/predictor_countdown/README.md) |
| Useful one-step prediction | Y + mask + position was selected by development accuracy | [Input ablation](../predictor/countdown_predictor_results/README.md#one-step-input-selection) |
| Multi-step calibration | Complete three-seed, 2,048-question evaluation through 32 single-layer updates | [Current calibration report](../predictor/countdown_predictor_results/README.md#dynamic-calibration) |
| All-layer prediction | Independent/shared architectures explored; full downstream experiment unfinished | Pending |

## Next experiments

1. **Isolate epoch selection.** Match predictor optimizer-state policy between fixed-5 and validation-selected calibration. The completed comparison changes both factors.
2. **Finish the all-layer extension.** Compare 24 independent predictors with a layer-conditioned shared predictor using gradient development data; complete one-step and 16-step calibration/oracle/frozen downstream tests for all layers' `o_proj`. Architecture fitting and downstream benefit are separate questions.
3. **Measure remaining drift and cost.** Track activation and aggregate A/B cosine, norm ratio, and error alongside accuracy. Count gradient labels and computation for calibration, validation, and diagnostics separately.

These are research priorities, not claims of completed results or instructions to launch a new run. Longer step counts and all-layer performance require their own matched evidence.

## Evidence conventions

- Use independent calibration-training and gradient-validation roles; do not choose epochs using downstream test accuracy.
- Record the input configuration, model state, update scope, seed pairs, optimizer reset policy, and actual selected epochs.
- Preserve full evaluation inputs; distinguish output-generation caps from input length.
- Separate gradient reconstruction, downstream task accuracy, and computational benefit.
- Update the main topic report for related variants; keep per-run configurations and raw tables as supporting evidence.

The [original data-selection plan](DATA_SELECTION_PLAN.md) and [SFT research archive](../research/README.md) remain available as earlier context.
