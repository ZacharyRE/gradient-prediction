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
| Joint vs independent updates | Joint adaptive 8.04% < warmup 8.64%; oracle 10.73%. Independent 3/8/19 tests complete | [Layer comparison](../predictor/countdown_predictor_results/README.md#joint-and-independent-layer-updates) |
| Predictor history transfer | Lower gradient error in 24/24 matched probes; carry 6.93% vs reset 6.12%, unchanged model 7.00%, on fresh test1024 | [Matched history study](../predictor/countdown_predictor_results/predictor-history-transfer/README.md) |
| All-layer prediction | Expansion deferred pending joint-update diagnosis | No completed downstream result |

## Next experiments

1. **Diagnose the joint-update gap.** Independent layer 3 shows large seed variation and a larger predictor/oracle gap. Test mixed predictor/oracle updates from the same jointly warmed checkpoint to isolate layer contributions; independent single-layer results do not identify the joint failure's cause.
2. **Isolate epoch selection.** Match predictor optimizer-state policy between fixed-5 and validation-selected calibration. The completed comparison changes both factors.
3. **Use the recorded drift diagnostics and measure cost.** Activation and aggregate A/B cosine, relative L2, and norm ratios are already available at every update. Count calibration, validation, and diagnostic gradient labels separately before making efficiency claims.
4. **Revisit all-layer expansion after a useful joint-update result.** Further tuning must use development data; an independent confirmation set is needed after decisions informed by the existing test results.

These are proposed research priorities, not newly launched runs. The 32-step independent and three-layer experiments are complete; all-24-layer downstream performance remains unestablished.

## Evidence conventions

- Use independent calibration-training and gradient-validation roles; do not choose epochs using downstream test accuracy.
- Record the input configuration, model state, update scope, seed pairs, optimizer reset policy, and actual selected epochs.
- Preserve full evaluation inputs; distinguish output-generation caps from input length.
- Separate gradient reconstruction, downstream task accuracy, and computational benefit.
- Update the main topic report for related variants; keep per-run configurations and raw tables as supporting evidence.

The [original data-selection plan](DATA_SELECTION_PLAN.md) and [SFT research archive](../research/README.md) remain available as earlier context.
