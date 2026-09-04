#!/usr/bin/env python3
"""Evaluate base and LoRA SFT checkpoints on GSM8K Test and MATH-500."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradient_geometry.extraction import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def is_correct(dataset: str, row: dict, prediction: str) -> bool:
    # Parsing the full reference solution is more reliable than parsing the bare
    # MATH answer: all reference solutions contain an anchored boxed answer.
    if dataset.startswith("math"):
        gold = parse(row["solution"])
    else:
        gold_text = row["answer"].rsplit("####", 1)[-1].strip()
        gold = parse(gold_text)
    target = parse(prediction)
    return bool(gold and target and verify(gold, target))


def completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = load_jsonl(path)
    if any(row.get("index") != i for i, row in enumerate(rows)):
        raise RuntimeError(f"Non-contiguous cached predictions in {path}")
    return rows


def evaluate_one(
    llm: LLM,
    tokenizer,
    sampling: SamplingParams,
    dataset_name: str,
    rows: list[dict],
    variant_name: str,
    lora_request: LoRARequest | None,
    output_root: Path,
    chunk_size: int,
) -> dict:
    variant_dir = output_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = variant_dir / f"{dataset_name}_predictions.jsonl"
    summary_path = variant_dir / f"{dataset_name}_summary.json"
    cached = completed_rows(prediction_path)
    if len(cached) > len(rows):
        raise RuntimeError(f"Too many cached predictions in {prediction_path}")
    started = time.time()
    for start in range(len(cached), len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        prompts = [make_prompt(tokenizer, row["problem"] if dataset_name.startswith("math") else row["question"]) for row in chunk]
        outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=True)
        with prediction_path.open("a", encoding="utf-8") as handle:
            for offset, (row, output) in enumerate(zip(chunk, outputs)):
                generated = output.outputs[0]
                prediction = generated.text
                record = {
                    "index": start + offset,
                    "prediction": prediction,
                    "correct": is_correct(dataset_name, row, prediction),
                    "finish_reason": generated.finish_reason,
                    "generated_tokens": len(generated.token_ids),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    predictions = completed_rows(prediction_path)
    correct = sum(int(row["correct"]) for row in predictions)
    summary = {
        "status": "passed" if len(predictions) == len(rows) else "incomplete",
        "variant": variant_name,
        "dataset": dataset_name,
        "samples": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "max_new_tokens": sampling.max_tokens,
        "temperature": sampling.temperature,
        "truncated_generations": sum(row["finish_reason"] == "length" for row in predictions),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in predictions) / len(predictions),
        "elapsed_this_invocation_seconds": time.time() - started,
        "prediction_file": str(prediction_path.resolve()),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "result/MIXED/Qwen2.5-1.5B-Instruct/sft_training/all_linear_r32_alpha64",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "result/MIXED/Qwen2.5-1.5B-Instruct/sft_evaluation/all_linear_r32_alpha64",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument(
        "--math-jsonl",
        type=Path,
        help="Evaluate only this MATH-format JSONL, under the dataset name mathtrain1000.",
    )
    args = parser.parse_args()

    adapters = {
        "base": None,
        "selected_epoch1": args.training_root / "selected/checkpoint-epoch-1",
        "selected_epoch2": args.training_root / "selected/checkpoint-epoch-2",
        "selected_epoch3": args.training_root / "selected/checkpoint-epoch-3",
        "random_epoch1": args.training_root / "random/checkpoint-epoch-1",
        "random_epoch2": args.training_root / "random/checkpoint-epoch-2",
        "random_epoch3": args.training_root / "random/checkpoint-epoch-3",
    }
    if args.variants:
        unknown = set(args.variants) - set(adapters)
        if unknown:
            raise ValueError(f"Unknown variants: {sorted(unknown)}")
        adapters = {name: adapters[name] for name in args.variants}
    for name, path in adapters.items():
        if path is not None and not (path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Missing adapter for {name}: {path}")

    if args.math_jsonl:
        datasets = {"mathtrain1000": load_jsonl(args.math_jsonl)}
    else:
        datasets = {
            "gsm8k": load_jsonl(ROOT / "data/GSM8K/test.jsonl"),
            "math500": load_jsonl(ROOT / "data/MATH-500/test.jsonl"),
        }
    if args.limit:
        datasets = {name: rows[: args.limit] for name, rows in datasets.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        seed=42,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_model_len=3072,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=42)
    summaries = []
    for lora_id, (variant_name, adapter_path) in enumerate(adapters.items(), start=1):
        request = None if adapter_path is None else LoRARequest(variant_name, lora_id, str(adapter_path.resolve()))
        for dataset_name, rows in datasets.items():
            summaries.append(
                evaluate_one(
                    llm, tokenizer, sampling, dataset_name, rows, variant_name,
                    request, args.output, args.chunk_size,
                )
            )
    overall = {
        "status": "passed" if all(row["status"] == "passed" for row in summaries) else "incomplete",
        "model": str(args.model.resolve()),
        "grading": {
            name: (
                "math_verify equivalence against boxed answer in reference solution"
                if name.startswith("math")
                else "math_verify equivalence against official #### final answer"
            )
            for name in datasets
        },
        "generation": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens, "seed": 42},
        "results": summaries,
    }
    write_json(args.output / "summary.json", overall)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
