# Predictor → training surrogate: results at a glance

**Positive result:** the new early-use CV16 recipe improves accuracy over the original CV16. **Remaining gap:** it does not beat the best cheap Oracle controls on time versus quality, and reliable behavior prediction is not yet demonstrated.

## Setup

Qwen2.5-1.5B-Instruct · MNLI · one H200 · single-layer LoRA (block 8, attention output projection, rank/alpha 64). Shared 32-update warmup; batch 32; AdamW. Final results average **3 new starts × 2,049 heldout examples** from the MNLI training population, not official validation. Full evaluation inputs are preserved. No new 0.5B/7B results in this study.

**CVk:** predict gradients for all 32 examples, correct using k true gradients. **Oraclek:** use k true gradients without a predictor.

## Main heldout comparison

Time = preparation + additional updates; excludes common warmup, evaluation and offline predictor construction. Each start uses the median of three exact-weight timing replays.

| Method | Additional updates | LR | Accuracy | Time |
|---|---:|---:|---:|---:|
| Warmup baseline | 0 | — | 77.45% | 0 s |
| Original CV16 | 256 | 0.001 | 82.43% | 20.76 s |
| **State CV16 for 32 updates → Oracle16** | 256 | 0.001 | **84.14%** | **19.46 s** |
| Matched Oracle16 | 256 | 0.001 | 83.78% | 18.86 s |
| Short Oracle32 | 128 | 0.001 | 83.81% | 15.61 s |
| Primary Oracle32 | 256 | 0.001 | 83.72% | 27.38 s |
| Stronger Oracle32, graph backend | 256 | 0.0003 | 84.35% | 30.06 s |
| Tuned original CV16 | 512 | 0.0003 | 84.69% | 37.44 s |
| Tuned Oracle16 | 512 | 0.0003 | 84.50% | 34.13 s |
| Adam, true gradient every 4 updates | 256 | 0.001 | 83.37% | 10.17 s |

- **Internal improvement:** early-use State CV16 gains **+1.71 percentage points** over original CV16 (paired 95% CI **+0.26 to +3.34**).
- **Oracle advantage remains unproven:** versus matched Oracle16, the gain is +0.36 points (CI −0.30 to +0.95), with slightly more time.
- **Beyond 32 updates remains unresolved:** the successful hybrid stops using the predictor after update 32.
- **Construction matters:** recorded offline preparation is 124.0 s for the original predictor and 488.8 s for State256. These historical component totals are additional, amortizable costs.

## CV4 was also tested

Development only: 3 starts, 513 examples, 512 updates, LR 0.0003, graph backend. These scores are **not directly comparable to the heldout table above**.

| Method | Accuracy | Prep + updates |
|---|---:|---:|
| CV4 | **82.85%** | 18.97 s |
| Matched Oracle4 | 81.35% | **15.88 s** |

CV4 gains 1.50 points over Oracle4, but adds time. This is promising development evidence, not independent final confirmation.

## Did it become a useful surrogate?

| Question | Evidence | Conclusion |
|---|---|---|
| Less compute? | Candidate0: CV16/128 = 1.40 matrix PFLOPs / 10.20 s; short Oracle32/64 = 1.48 / 7.76 s | Fewer FLOPs did not imply lower latency. |
| Better data selection? | CV16/generated scoring gains +0.49 / +0.57 points over random choice for random / genre candidates; both CIs include zero | No established selection advantage. |
| Faster selection workflow? | CV16 + learned behavior: 133.65 s; short Oracle + exact scoring: 123.93 s | No end-to-end speed win; offline fitting is extra. |
| Predict the trained agent's behavior? | Learned behavior: 76.96%; unchanged initial predictions: 77.62%; Oracle agent: 83.72% (forced ABC metric) | Changed-agent behavior is not captured reliably. |

State conditioning, additional training states/data, sparse calibration, token critics, copied teacher blocks and gradient memory were explored. None established a general time–quality win. [Attempts and failures](DEVELOPMENT.md) · [Late experiments](LATE_DEVELOPMENT.md).

**Next experiment:** compare CV4/CV16 hybrids with short Oracle at equal total time on new starts. Require sustained late-state usefulness before claiming long-horizon acceleration. For data selection, test direct prediction of training-data rankings.

[Complete tables, FLOPs and intervals](RESULTS.md) · [Detailed study](STUDY_DETAILS.md) · [Methods](METHODS.md) · [Reproduction and cache availability](REPRODUCING.md)
