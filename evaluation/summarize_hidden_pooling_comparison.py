#!/usr/bin/env python3
"""Compare mean-pooled prompt hidden states with the last prompt token."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "predictor" / "single_layer" / "results"


def load_summary(name: str) -> dict:
    path = RESULTS / name / "formal_2000_1000" / "ridge" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for rank in (2, 4):
        for layer in (5, 11):
            for short in ("q", "v", "o"):
                stem = f"layer{layer}_{short}proj_rank{rank}"
                last = load_summary(f"{stem}_direct")
                mean = load_summary(f"{stem}_mean_hidden_direct")
                last_result = last["results"]["ridge_test"]
                mean_result = mean["results"]["ridge_test"]
                last_factors = last["results_by_lora_factor"]
                mean_factors = mean["results_by_lora_factor"]
                last_baseline = last["results"]["mean_gradient_test"]["mean_cosine"]
                mean_baseline = mean["results"]["mean_gradient_test"]["mean_cosine"]
                rows.append({
                    "rank": rank,
                    "layer": layer,
                    "module": f"{short}_proj",
                    "baseline_last": last_baseline,
                    "baseline_mean": mean_baseline,
                    "baseline_absolute_difference": abs(last_baseline - mean_baseline),
                    "last_cosine": last_result["mean_cosine"],
                    "mean_cosine": mean_result["mean_cosine"],
                    "mean_minus_last_cosine": (
                        mean_result["mean_cosine"] - last_result["mean_cosine"]
                    ),
                    "last_a_cosine": last_factors["A"]["ridge_test"]["mean_cosine"],
                    "mean_a_cosine": mean_factors["A"]["ridge_test"]["mean_cosine"],
                    "last_b_cosine": last_factors["B"]["ridge_test"]["mean_cosine"],
                    "mean_b_cosine": mean_factors["B"]["ridge_test"]["mean_cosine"],
                    "last_target_spearman": last["target_alignment_ranking"]["spearman"],
                    "mean_target_spearman": mean["target_alignment_ranking"]["spearman"],
                    "mean_minus_last_target_spearman": (
                        mean["target_alignment_ranking"]["spearman"]
                        - last["target_alignment_ranking"]["spearman"]
                    ),
                })
    cosine_deltas = [row["mean_minus_last_cosine"] for row in rows]
    target_deltas = [row["mean_minus_last_target_spearman"] for row in rows]
    summary = {
        "protocol": {
            "layers": [5, 11],
            "modules": ["q_proj", "v_proj", "o_proj"],
            "ranks": [2, 4],
            "comparison": "mean_prompt_tokens versus last_prompt_token",
            "predictor_train_samples": 2000,
            "held_out_test_samples": 1000,
        },
        "aggregate": {
            "mean_cosine_delta": sum(cosine_deltas) / len(cosine_deltas),
            "mean_pooling_cosine_wins": sum(delta > 0 for delta in cosine_deltas),
            "last_token_cosine_wins": sum(delta < 0 for delta in cosine_deltas),
            "mean_target_spearman_delta": sum(target_deltas) / len(target_deltas),
            "mean_pooling_target_wins": sum(delta > 0 for delta in target_deltas),
            "last_token_target_wins": sum(delta < 0 for delta in target_deltas),
            "maximum_baseline_absolute_difference": max(
                row["baseline_absolute_difference"] for row in rows
            ),
        },
        "rows": rows,
    }
    json_path = RESULTS / "hidden_pooling_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Prompt hidden-state pooling comparison",
        "",
        "| Rank | Layer | Module | Baseline | Last cosine | Mean cosine | Delta | Last target rho | Mean target rho | Delta |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['layer']} | {row['module']} | "
            f"{row['baseline_last']:.4f} | {row['last_cosine']:.4f} | "
            f"{row['mean_cosine']:.4f} | {row['mean_minus_last_cosine']:+.4f} | "
            f"{row['last_target_spearman']:.4f} | {row['mean_target_spearman']:.4f} | "
            f"{row['mean_minus_last_target_spearman']:+.4f} |"
        )
    markdown_path = RESULTS / "hidden_pooling_comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


if __name__ == "__main__":
    main()
