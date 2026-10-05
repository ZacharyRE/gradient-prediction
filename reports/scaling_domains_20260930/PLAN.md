# GPU0 / 24-hour experiment, 2026-09-30 Phoenix

User authorization: 24 hours, physical GPU0 only (H200), ending 2026-10-02 04:19:12 UTC / 2026-10-01 21:19:12 Phoenix. Preserve complete training/evaluation inputs. No input truncation. Output generation budgets are separate. Produce a final Chinese report with all attempts, configurations, insights, failures, statistical and computational limits.

## Questions and priorities

1. Can gentler warmup and separately tuned subsequent updates establish an SFT baseline that improves on the unadapted model and/or warmup start?
2. Does activation-gradient prediction preserve useful adaptation across mathematics, algebra, elementary science, and reading comprehension? Which parts depend on warmup, model scale, update step size, and calibration frequency?
3. Do frozen or periodically calibrated predictors save actual single-GPU time and arithmetic work? Account for label collection, calibration selection, offline fitting, and quality, not only backbone backward calls.

## Scope fixed before new results

Nine task configurations from eight dataset families: Countdown, GSM8K, MATH-500, SVAMP, AQuA-RAT, ARC-Easy, ARC-Challenge, OpenBookQA, BoolQ. ARC-Easy/Challenge are not independent domains. New source revisions and source hashes are frozen in data/manifest.json. Existing three mathematics tests are historical benchmarks, not pristine new tests. BoolQ official validation becomes this study's final holdout; all tuning data comes from official training. No final test scores/text are used for tuning. All data splits are normalized-question disjoint within task; cross-task overlap is audited separately and models are separately fitted.

Primary breadth model: Qwen2.5-1.5B-Instruct. Qwen2.5-7B-Instruct provides paired scale checks, prioritized on Countdown, GSM8K, AQuA and at least one science task if resources permit. Both use a single layer8 o_proj rank=alpha64 adapter initially. Model size is not an isolated scaling-law experiment.

Warmup pilot: common seed101, fixed examples/order, AdamW, batch32, clip1, weight decay .01; LR {0,3e-6,1e-5,5e-5,1e-4,5e-4}; save 8/32-step states and the zero-step initialization. Zero LR is represented by zero steps, not redundant identical updates. Evaluate generation development accuracy and teacher-forced CE. Keep all outcomes including degradation. Select one nonzero warmup by dev accuracy, then lower CE, then fewer steps/lower LR; no-warmup remains an explicit control and may be selected as a separate sensitivity condition. Final SFT and predictor comparisons include Base and Warmup.

Main update pilot: 32 updates, AdamW weight decay0, batch32, clipping1, new optimizer from the common start. LR grid {3e-6,1e-5,3e-5,1e-4}; stop-at-start is a reported baseline. Select fixed step32 LR independently by development generation for each method. Earlier 8/16 checkpoints are sensitivity curves, not post-test endpoint selection. If development evidence requires new candidates, add an explicit amendment before final selection/final evaluation.

Predictor: original width512 depth2 bidirectional Transformer, Y+mask+position+logRMS, factor-balanced A/B error plus .25 normalized activation error, no direct parameter head. Primary offline budget up to2048 distinct training questions (smaller for small datasets), 100 epochs with gradient-dev checkpoint choice. The changed budget relative to old4096/8192 experiments must be disclosed. Compare Frozen, Adaptive every-step 4+16 labels, periodic calibration every8 steps, matched-label Mean, Oracle32, and fresh-subset Oracle20. Periodic calibration is a cost/quality sensitivity; do not assume it helps. At zero initialization B=0, A gradients are structurally zero and must be reported separately.

## Selection and evidence

Each task reserves predictor-dev, gradient-audit, update, and generation-dev separately. Full official test sets are retained. Split sizes and all sampling seeds are recorded; no example is selected/excluded by length. Short-answer science and boolean tasks request a boxed answer without fabricated rationale; mathematics retains available solutions/rationales. Prompt/target style differences are part of the task setting, not a pure domain-only intervention.

Final candidate manifests freeze hashes and development choices before final evaluation. Pair comparisons on the same questions. Report absolute accuracy, changes versus Base/Warmup, versus same-LR and independently tuned Oracle/Mean; paired bootstrap intervals and a declared primary multiplicity family. These are conditional on fitted runs. Replicate representative settings with predeclared seeds124/102 and125/103 as budget allows, without selecting seeds by final outcomes. Any incomplete experiment is labeled incomplete.

## Efficiency

Use GPU0 only; synchronized timings after warmup, randomized method order, identical model/optimizer start and batch. Separate diagnostic backward work from deployed-method work. Profile actual tensor operations and document counted/unsupported operations, attention implementation, arithmetic convention, and precision. Report profiler FLOPs as measured operator coverage, not a fictitious complete hardware counter. Report peak allocated memory, per-stage time, processed tokens, true-gradient labels, and offline costs. Efficiency claims require quality comparison; include accuracy versus cumulative update time checkpoints. Search/evaluation costs are research costs, shown separately from a single deployment. Do not promise speedup before measurement.

## Resource and storage policy

Serial GPU workers. No other GPU is used or other user's process stopped. Keep >45GiB free; caches are processed per condition, then removed only after downstream dependent fits/audits succeed. Retain source manifests, final/selected adapters and predictors, raw full generation results, histories and summaries. Never remove older experiments. Stop launching nonessential searches with at least4 hours left; reserve time for frozen final evaluation, audits and report. Runtime-based scope changes are logged independently of scores. Daily-goal completion requires the final report and evidence, not just launching workers.
