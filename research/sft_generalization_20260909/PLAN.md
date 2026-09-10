# LoRA SFT generalization investigation

Start: 2026-09-09 15:16:40 UTC. Hard deadline: 2026-09-10 15:16:40 UTC.
Only GPUs 1 and 3 may be used. No changes to previous experiments or main training code.

Question: why does Qwen2.5-1.5B-Instruct remain near baseline after the previous target replacement repair, and which intervention can yield reproducible improvement?

Prior evidence: raw reference SFT degraded generation; greedy-correct self-target replacement repaired regression on matched questions, without significant gain. Mask/shift/token accumulation, rank/modules/full updates, attention arithmetic, DFT, and extended original-data training were already examined. Reuse these results rather than repeating their searches.

Primary hypotheses, recorded before new outcomes:

1. Greedy-correct-only supervision excludes correct trajectories for problems the student currently gets wrong. Test sampled-and-verified student solutions and verified stronger-teacher solutions, with matched question subsets and explicit difficulty/selection accounting.
2. Target length and effective per-example weighting interfere with transfer. Compare token-mean and example-mean on identical target files; do not call this a code bug, since token-mean is a valid objective.
3. Supervision scale/diversity and optimization strength are insufficient after distribution repair. Examine finite LR/epoch settings on development only, and expand data only if smaller controlled tests warrant it.
4. Off-distribution target pressure causes forgetting. Use matched target-source and (if needed) retention controls; low CE alone never establishes learning of answer accuracy.

Phase A: inventory existing artifacts; generate n=8 student samples at temperature 0.7 on the original 2000 training problems; generate 7B teacher solutions on the same problems. No evaluation questions or answers may enter target construction. Preserve every output and filter decision. Strict final-box scoring and complete untruncated sequences are required for candidate targets; final answers do not certify intermediate reasoning.

Phase B: seed43 exploratory development matrix. Primary development dataset is the existing MATH dev500. Historical MATH confirmation sets are now reused validation, never fresh holdouts. Select one main recipe and its matched controls before accessing newly reserved OOD confirmation performance. All negative branches must be retained.

Phase C: seeds43/44/45 from the fixed base, identical recipe. Main improvement claim requires every seed to exceed baseline on the prespecified primary math evaluation and a paired across-question mean difference whose 95% CI excludes zero. Report per-seed CIs, crossed seed/question resampling, and outcome changes. New mathematical OOD benchmarks provide independent task-transfer evidence; cross-benchmark macro improvement and individual dataset outcomes are reported, without calling mathematical transfer universal general capability. OOD confirmation is held out from recipe selection. If used adaptively it must be relabeled development and replaced.

For a setup explanation, require a controlled intervention, replicated direction, and supporting mechanistic diagnostics. A successful data-source bundle identifies that bundle, not every component. A failure to improve does not prove universal LoRA/SFT impossibility. At the 24-hour deadline stop new compute, retain unfinished evidence, and give a precise supported or unresolved answer.

Deliverables: comprehensive Chinese REPORT.md and rendered report, source notes, protocols/hashes, raw predictions, training histories/adapters, paired statistics, figures with CSV data, reproduction commands, experiment inventory and resource/time audit.


## Prospective precision amendment, 2026-09-09 19:32 UTC

Before any new1500-test or OOD predictions, final mathematical evaluation is expanded to the full5000 official MATH test, with unchanged greedy2048 protocol andall3seeds/pairedCI requirements. Maxprompt1384 fits existing4096context. Both1500 andfull5000 are historically reused validation, never claimedfresh. The1500 subset remains developmental validation; final5000 is evaluated only for frozenselectedrecipe anddeclaredcontrols. Also report4997 sensitivity excluding3 conservative alphanumeric-match candidates; reviewfoundzero exactwhitespace-normalizedtrain duplicates andtwo falsematches causedby removingminus signs. Thethird is cosine-evenness equivalent. OOD878 andprimaryequal-weight3datasetmacroremain untouchedandunchanged. See audits/primary_math_precision_amendment.json andfull_math_overlap_review.json. Earlierdevfailures remain partofevidence; no endpoint waschanged afternewtestoutcomes.

## Final-seed precision amendment (2026-09-09 21:21 UTC)

Before any reserved OOD or full5000 result, final primary confirmation is strengthened to seeds43/44/45/46/47. Allfive must exceed the fixed baseline on primaryMATH and OOD macro; both prespecified mean confidence intervals must exclude zero. Existing three-seed development analyses are preserved. No finalrecipe has been selected and no confirmation score observed. See audits/final_seed_precision_amendment.json. Review the exploration queue at06:30UTC to reserve replication/evaluation/report time under the original24h limit. Do not drop unfavorable seeds.
