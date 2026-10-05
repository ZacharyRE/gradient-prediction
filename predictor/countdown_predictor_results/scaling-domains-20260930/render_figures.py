"""Rebuild the release figure from published CSVs: python render_figures.py."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / "results" / name).open() as f:
        return list(csv.DictReader(f))


rows = read("main_comparison.csv")
profiles = read("controlled_efficiency.csv")
names = {"countdown": "Countdown", "svamp": "SVAMP", "aqua": "AQuA",
         "gsm8k": "GSM8K", "math": "MATH", "arc_easy": "ARC-Easy",
         "arc_challenge": "ARC-Challenge", "openbookqa": "OpenBookQA", "boolq": "BoolQ"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, cost) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.5, 1]})
y = np.arange(len(rows))
for field, label, color, shift in [
    ("oracle_minus_warmup_pp", "Oracle", "#64748b", -.12),
    ("adaptive_minus_warmup_pp", "Adaptive", "#2563eb", .12),
]:
    ax.scatter([float(r[field]) for r in rows], y + shift, label=label, color=color, s=36)
ax.axvline(0, color="#94a3b8", linewidth=1)
ax.axhline(8.5, color="#e2e8f0", linewidth=1)
ax.set_yticks(y, [("1.5B" if r["model"] == "1p5b" else "7B") + "  " + names[r["task"]] for r in rows])
ax.invert_yaxis()
ax.set_xlim(-2.8, 3.2)
ax.set_xlabel("Accuracy change from warmup (percentage points)")
ax.set_title("32 more updates: limited additional benefit", loc="left", fontsize=12, pad=14)
ax.legend(loc="lower left", frameon=False, ncol=2)
ax.grid(axis="x", alpha=.12)

for i, (method, label) in enumerate([("frozen", "Frozen"), ("adaptive", "Adaptive")]):
    subset = [r for r in profiles if r["method"] == method]
    for field, offset, color, metric in [
        ("time_ratio_to_oracle", -.15, "#d97706", "Wall time"),
        ("flop_ratio_to_oracle", .15, "#0891b2", "Counted FLOPs"),
    ]:
        lo, hi = min(float(r[field]) for r in subset), max(float(r[field]) for r in subset)
        cost.plot([lo, hi], [i + offset] * 2, color=color, linewidth=7,
                  solid_capstyle="round", label=metric if i == 0 else None)
        cost.text(hi + .07, i + offset, f"{lo:.2f}–{hi:.2f}×", va="center", fontsize=10)
cost.axvline(1, color="#94a3b8", linestyle="--", linewidth=1)
cost.text(1.04, -.42, "Oracle = 1×", color="#64748b", fontsize=9)
cost.set_yticks([0, 1], ["Frozen", "Adaptive"])
cost.set_ylim(1.65, -.65)
cost.set_xlim(0, 3.85)
cost.set_xlabel("Ratio to Oracle (lower is cheaper)")
cost.set_title("Fewer FLOPs do not guarantee less time", loc="left", fontsize=12, pad=14)
cost.legend(loc="lower right", frameon=False)
cost.grid(axis="x", alpha=.12)
fig.text(.02, .015, "Left: fixed-run point estimates; each method selects LR on dev. Right: ranges across 13 H200 profiles; offline costs excluded.", fontsize=9, color="#475569")
fig.tight_layout(rect=(0, .05, 1, 1), w_pad=3)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "overview.png", dpi=180)
plt.close(fig)
