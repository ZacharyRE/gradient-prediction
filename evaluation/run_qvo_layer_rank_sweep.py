#!/usr/bin/env python3
"""Run the single-layer q/v/o direct-gradient sweep with resumable stages."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
LAYERS = (5, 11, 23, 27)
MODULES = ("q_proj", "v_proj", "o_proj")
RANKS = (4, 2)


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in {
            "passed", "complete"
        }
    except (json.JSONDecodeError, OSError):
        return False


def experiment_name(layer: int, module: str, rank: int, pooling: str) -> str:
    short = module.removesuffix("_proj")
    pooling_suffix = "_mean_hidden" if pooling == "mean_prompt_tokens" else ""
    return f"layer{layer}_{short}proj_rank{rank}{pooling_suffix}_direct"


def make_config(layer: int, module: str, rank: int, pooling: str) -> Path:
    short = module.removesuffix("_proj")
    template = ROOT / "configs" / "gradient_geometry" / (
        f"qwen2.5_1.5b_layer17_{short}proj_rank4_direct.yaml"
    )
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    name = experiment_name(layer, module, rank, pooling)
    config["experiment"]["name"] = f"qwen2.5_1.5b_{name}_gradient"
    config["model"]["hidden_layer_zero_based"] = layer
    config["model"]["hidden_states_tuple_index"] = (
        layer + 1 if module == "o_proj" else layer
    )
    config["model"]["hidden_token"] = pooling
    if config["model"].get("hidden_source") == "target_module_input":
        config["model"]["hidden_module_suffix"] = (
            f"layers.{layer}.self_attn.{module}"
        )
    config["lora"]["rank"] = rank
    # Preserve the alpha/rank scale of the rank-4, alpha-8 experiments.
    config["lora"]["alpha"] = rank * 2
    config["lora"]["target_modules"] = [module]
    config["lora"]["target_layers"] = [layer]
    config["predictor"]["model"] = ["ridge"]
    output = ROOT / "configs" / "gradient_geometry" / f"qwen2.5_1.5b_{name}.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def run_stage(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_job(job: tuple[int, str, int], gpu: str, pooling: str) -> str:
    layer, module, rank = job
    name = experiment_name(layer, module, rank, pooling)
    config = ROOT / "configs" / "gradient_geometry" / (
        f"qwen2.5_1.5b_{name}.yaml"
    )
    base = ROOT / "predictor" / "single_layer" / "results" / name
    warmup = base / "warmup_128_steps32"
    formal = base / "formal_2000_1000"
    log = base / "sweep.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    python = str(ROOT / ".venv" / "bin" / "python")

    if not complete(warmup / "summary.json"):
        run_stage(
            [python, "training/warmup_lora.py", "--config", str(config),
             "--output", str(warmup), "--device", "cuda:0"],
            log, env,
        )
    if not complete(formal / "extraction_summary.json"):
        run_stage(
            [python, "evaluation/extract_single_layer_raw_gradients.py",
             "--config", str(config), "--adapter", str(warmup / "adapter"),
             "--output", str(formal), "--device", "cuda:0"],
            log, env,
        )
    if not complete(formal / "ridge" / "summary.json"):
        run_stage(
            [python, "evaluation/evaluate_direct_raw_gradient_ridge.py",
             "--experiment", str(formal), "--device", "cuda:0"],
            log, env,
        )
    if not complete(formal / "target_alignment" / "summary.json"):
        run_stage(
            [python, "evaluation/evaluate_target_alignment_spearman.py",
             "--config", str(config), "--adapter", str(warmup / "adapter"),
             "--experiment", str(formal), "--device", "cuda:0"],
            log, env,
        )
    return name


def run_gpu_queue(
    gpu: str, jobs: list[tuple[int, str, int]], pooling: str
) -> tuple[list[str], dict[str, str]]:
    completed: list[str] = []
    failures: dict[str, str] = {}
    for job in jobs:
        name = experiment_name(*job, pooling)
        try:
            completed.append(run_job(job, gpu, pooling))
            print(f"COMPLETE {name} on GPU {gpu}", flush=True)
        except Exception as exc:  # Keep independent jobs running.
            failures[name] = repr(exc)
            print(f"FAILED {name} on GPU {gpu}: {exc!r}", flush=True)
    return completed, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["1", "3", "5"])
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    parser.add_argument("--ranks", nargs="+", type=int, default=list(RANKS))
    parser.add_argument(
        "--pooling",
        choices=("last_prompt_token", "mean_prompt_tokens"),
        default="last_prompt_token",
    )
    args = parser.parse_args()
    jobs = [
        (layer, module, rank)
        for rank in args.ranks for layer in args.layers for module in MODULES
    ]
    for job in jobs:
        make_config(*job, args.pooling)

    failures: dict[str, str] = {}
    completed: list[str] = []
    queues = [jobs[index::len(args.gpus)] for index in range(len(args.gpus))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(run_gpu_queue, gpu, queue, args.pooling)
            for gpu, queue in zip(args.gpus, queues, strict=True)
        ]
        for future in concurrent.futures.as_completed(futures):
            gpu_completed, gpu_failures = future.result()
            completed.extend(gpu_completed)
            failures.update(gpu_failures)

    report = {"completed": sorted(completed), "failures": failures}
    report_stem = (
        "qvo_layer_rank_mean_hidden_sweep"
        if args.pooling == "mean_prompt_tokens"
        else "qvo_layer_rank_sweep"
    )
    report_path = ROOT / "predictor" / "single_layer" / "results" / f"{report_stem}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
