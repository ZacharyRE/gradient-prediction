# Do better targets create useful Oracle adaptation?

**Complete — October 5, 2026.** Qwen2.5-1.5B-Instruct on GSM8K; only zero-indexed layer 8 `o_proj` LoRA, rank = alpha = 64. Every update uses true gradients. **No predictor was trained in this study.**

The main finding is a modest cumulative SFT signal, with **uncertain improvement after warmup**. Teacher supervision looks promising against the untrained model, but neither repeated-question nor fresh-question continuation establishes a reliable advantage over stopping after warmup.

All means below use three training seeds and the complete **1,319-question test**. Untrained **Base = 73.09%**.

| Supervision target | Dev-selected LR × updates | Test mean | Fixed `1e-5 × 32` mean |
|---|---:|---:|---:|
| Original reference | `3e-5 × 32` | 73.57% | 74.32% |
| Verified student-generated | `3e-4 × 32` | 73.82% | 74.32% |
| Verified 7B-teacher-generated | `1e-4 × 256` | **75.08%** | 73.92% |
| Reference with calculator spans removed | `3e-5 × 32` | 73.26% | 74.02% |

- **Teacher gain is promising, not confirmed:** +2.00 percentage points against Base; joint seed/question 95% CI **[−0.03, +4.02]**. The prespecified familywise stable-gain criterion is not met.
- **Smaller LR helps some targets:** reference and student reach the same 74.32% at the fixed smaller LR. Their nominal intervals exclude zero narrowly; familywise intervals do not. Self-generated targets are not necessary for this observed modest gain.
- **Generated targets tolerate the tested settings better:** development accuracy spans 17.58 points for reference, versus 3.52/3.91 for student/teacher. Lower training loss does not guarantee better downstream accuracy.

## The important baseline: stop after warmup

These are **post-hoc** diagnostics at teacher LR `1e-4`, with identical warmup parameters and retained optimizer state. “Stop” means no further model updates, not a frozen gradient predictor.

| After the same 32-update warmup | Total updates | Unique training questions seen | Test mean | Gain over stopping |
|---|---:|---:|---:|---:|
| Stop updating | 32 | 907 | 74.40% | — |
| Continue on the original pool | 256 | 907 | 75.08% | +0.68 pp |
| Continue on a disjoint new pool | 256 | 1,814 | 74.96% | +0.56 pp |

The continuation-gain 95% CIs are **[−1.21, +2.55]** and **[−1.16, +2.27]**. Fresh versus repeated questions is **−0.13 pp**, CI **[−1.79, +1.57]**. This is not evidence of equivalence or proof that more data cannot help. Only one new pool and one continuation recipe were tested; each pool still contains questions the student solved in its sampled attempts.

The original 128-update teacher checkpoint averages 75.41%, but this exploratory test curve does not replace the frozen, dev-selected primary recipe.

## Compute and scope

Dedicated median Oracle update times on an H200 are **0.490 / 0.567 / 0.576 seconds** for reference/student/teacher targets. Supported-operator FLOP estimates on one matched 32-question batch are **48.14 / 84.35 / 87.95 TFLOPs**. These are different measurement scopes: the FLOP profile is partial and does not estimate average dataset FLOPs. No predictor speedup is measured.

The completed work includes **36 original training trajectories, three fresh-data branches, 37 dev versions, and 40 full-test versions**. The report accounts separately for generation, search, exact replays, timing fits and evaluation. Initial replication timings affected by external GPU contention are identified; dedicated timings were collected separately.

Inputs and retained supervision are preserved in full. **2,048 tokens limits generated output only.** Targets are generated once and held fixed; this is not online answer regeneration. Calculator-span removal also removes internal expressions, so it is not a pure formatting control. Neither offline predictor sample-size scaling nor larger student models is tested here.

**Next priority:** establish reproducible Oracle improvement over a fixed warmup-only baseline on independently held-out, harder or shifted data. Then evaluate a predictor against both Oracle and no further updating, with wall time and FLOPs reported separately.

- [Full report: configurations, all results, uncertainty, costs and limitations](REPORT.md)
- [Accuracy overview](figures/overview.png) · [Development sensitivity](figures/all_target_development.png) · [Teacher continuation](figures/teacher_continuation.png)
- [Primary protocol](configs/protocol.json) · [Statistical plan](configs/statistics_plan.json) · [Fixed small-LR control](configs/lowlr_protocol.json)
- [Same-pool continuation protocol](configs/teacher_prefix_protocol.json) · [Fresh-question extension protocol](configs/fresh_protocol.json)
- [Complete attempt inventory](results/experiment_inventory.json) · [Completion audit](results/completion_audit.json)

Frozen inputs, raw generations, full test outputs, compact adapters, configurations, histories and analysis scripts are retained. No full model copies or activation caches were created. Rebuild the report from saved analysis results with:

```bash
../.venv/bin/python target_supervision_20261005/scripts/report_progress.py
```
