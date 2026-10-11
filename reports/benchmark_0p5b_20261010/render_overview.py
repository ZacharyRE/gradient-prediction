"""Render the ten admitted configurations from the published summary CSV."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "summary.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["predictor_admitted"] == "True"]
    assert len(rows) == 10
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.83, bottom=0.14)
    for i in range(len(rows)):
        if i % 2 == 0:
            ax.axhspan(i - 0.48, i + 0.48, color="#f3f5f8", zorder=0)
    for method, color, offset, label in [
        ("oracle", "#2563a6", -0.14, "Oracle"),
        ("predictor", "#d77924", 0.14, "Predictor"),
    ]:
        means = [float(r[f"{method}_gain_pp"]) for r in rows]
        lower = [m - float(r[f"{method}_ci95_low_pp"]) for m, r in zip(means, rows)]
        upper = [float(r[f"{method}_ci95_high_pp"]) - m for m, r in zip(means, rows)]
        ax.errorbar(means, [i + offset for i in range(len(rows))],
                    xerr=[lower, upper], fmt="o", color=color, capsize=3,
                    markersize=6, linewidth=1.5, label=label)
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set_yticks(range(len(rows)), [r["benchmark"] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-4.5, 16.5)
    ax.set_xticks([-4, 0, 4, 8, 12, 16])
    ax.set_xlabel("Final accuracy gain over fixed warmup (percentage points)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Oracle gains are substantial; predictor retention varies",
                 x=0.04, ha="left", y=0.98, fontsize=17, fontweight="bold")
    fig.text(0.04, 0.915,
             "Qwen2.5-0.5B-Instruct · 10 oracle-admitted configurations · 3 update seeds",
             fontsize=11, color="#475569")
    fig.text(0.04, 0.035,
             "Nominal 95% paired intervals; conditional on warmup seed 201. Full evaluation inputs preserved.",
             fontsize=9, color="#475569")
    fig.savefig(ROOT / "overview.png", dpi=180)
    fig.savefig(ROOT / "overview.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
