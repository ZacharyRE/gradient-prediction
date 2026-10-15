# What to test next

These are proposed follow-ups, **not completed experiments**. The present study has stopped method selection and uses its frozen readouts for reporting.

1. **Verify the useful operating point on new starting adapters.** Retain CV16 and online class RLS8 as practical gradient-surrogate baselines. Use independently tuned Oracle32 stopping points, Oracle16/8, and gradient hold as controls. Repeat from new warmup seeds on previously unused evaluation groups. Require both acceptable downstream loss and a measured time advantage, including predictor preparation when reuse is unavailable.

2. **Change what the next surrogate learns before expanding another calibration grid.** Test a compact state-conditioned model that directly outputs a structured *per-example LoRA-gradient residual* around the online class baseline. Compare it with that baseline alone. Centered loss objectives were already screened in this study; the new hypothesis is the direct structured residual output and its combination with the online baseline. Simply adding more teacher states was also screened without an established downstream gain.

3. **Gate expensive downstream sweeps on late-state prediction.** Collect training labels from both teacher trajectories and the surrogate's own visited states, charging collection and teacher-training costs. Hold out complete trajectories/starting adapters. Inspect centered residual variance at 0/32/128/256/512 updates, then evaluate the actual updated model at fixed wall-time budgets. A scalar coefficient, an improved average-gradient cosine, or a good step-32 checkpoint is insufficient. Use delayed calibration so a correction sample cannot train its own current-step predictor.

For CV4, a residual/true variance ratio of 0.429 would match Oracle8's raw within-batch sampling noise at batch size 32; 0.143 would match Oracle16. Those are diagnostic targets, not downstream guarantees or mandatory accuracy thresholds. The original predictor is close to a ratio of 1 at late teacher states. A substantially better residual model is therefore a more specific research objective than merely increasing calibration frequency.

The wall-time test must include the whole update path. In the measured PAWS CV16 development run, true-gradient calls account for 16.74 of 18.69 online seconds, while prediction takes 1.49 seconds. Reducing predictor FLOPs alone has limited room to help. Skipping true-gradient calls is a separate hypothesis and needs strong stability controls: the completed random-query experiments include late collapses and weak gains over no-predictor reuse.

The current predictor is already a **surrogate for gradients or training updates**. It does not directly replace the trained agent's answers or actions. A behavior surrogate would require a different prediction target and an additional evaluation, which this study does not claim to provide.
