#!/usr/bin/env python3
"""Summarize completed q/v/o single-layer rank-sweep experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "predictor" / "single_layer" / "results"
PATTERN = re.compile(r"layer(\d+)_(q|v|o)proj_rank(2|4)_direct$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for experiment in RESULTS.iterdir():
        match = PATTERN.fullmatch(experiment.name)
        if not match:
            continue
        ridge_path = experiment / "formal_2000_1000" / "ridge" / "summary.json"
        target_path = experiment / "formal_2000_1000" / "target_alignment" / "summary.json"
        if not ridge_path.exists() or not target_path.exists():
            continue
        ridge = read_json(ridge_path)
        target = read_json(target_path)
        baseline = ridge["results"]["mean_gradient_test"]
        predicted = ridge["results"]["ridge_test"]
        factors = ridge["results_by_lora_factor"]
        rows.append({
            "layer": int(match.group(1)),
            "module": f"{match.group(2)}_proj",
            "rank": int(match.group(3)),
            "selected_alpha": ridge["selection"]["selected_alpha"],
            "combined_baseline_cosine": baseline["mean_cosine"],
            "combined_ridge_cosine": predicted["mean_cosine"],
            "combined_cosine_gain": predicted["mean_cosine"] - baseline["mean_cosine"],
            "combined_r2_variance_weighted": predicted["r2_variance_weighted"],
            "combined_norm_pearson": predicted["gradient_norm_pearson"],
            "a_baseline_cosine": factors["A"]["mean_gradient_test"]["mean_cosine"],
            "a_ridge_cosine": factors["A"]["ridge_test"]["mean_cosine"],
            "b_baseline_cosine": factors["B"]["mean_gradient_test"]["mean_cosine"],
            "b_ridge_cosine": factors["B"]["ridge_test"]["mean_cosine"],
            "target_spearman": target["results"]["ridge_shared_alpha"]["spearman"],
            "result_directory": str(experiment.relative_to(ROOT)),
        })
    rows.sort(key=lambda row: (row["rank"], row["layer"], row["module"]))

    summary = {
        "protocol": {
            "predictor_train_samples": 2000,
            "held_out_test_samples": 1000,
            "target_validation_samples": 500,
            "warmup_samples": 128,
            "warmup_steps": 32,
            "layers_newly_swept": [5, 11, 23, 27],
            "existing_rank4_reference_layer": 17,
            "modules": ["q_proj", "v_proj", "o_proj"],
            "ranks": [2, 4],
            "lora_alpha_over_rank": 2,
        },
        "num_completed_experiments": len(rows),
        "num_newly_completed_experiments": sum(row["layer"] != 17 for row in rows),
        "rows": rows,
    }
    json_path = RESULTS / "qvo_layer_rank_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Q/V/O single-layer rank sweep",
        "",
        "| Rank | Layer | Module | Baseline | Ridge | Gain | A Ridge | B Ridge | Target Spearman |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['layer']} | {row['module']} | "
            f"{row['combined_baseline_cosine']:.4f} | {row['combined_ridge_cosine']:.4f} | "
            f"{row['combined_cosine_gain']:+.4f} | {row['a_ridge_cosine']:.4f} | "
            f"{row['b_ridge_cosine']:.4f} | {row['target_spearman']:.4f} |"
        )
    markdown_path = RESULTS / "qvo_layer_rank_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
