#!/usr/bin/env python3
"""Create the fixed non-degenerate LoRA checkpoint for geometry experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.data import (  # noqa: E402
    build_fixed_splits,
    load_math500,
    load_math_train,
    public_manifest_row,
    select_warmup_rows,
)
from gradient_geometry.extraction import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_supervised_example,
    load_model_and_tokenizer,
    set_global_seed,
    trainable_lora_parameters,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_paths(config: dict) -> None:
    for key in ("math_train_path", "math500_path"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str(PROJECT_ROOT / path)


def collate_batch(tokenizer, rows: list[dict], config: dict, device: str):
    examples = [
        build_supervised_example(
            tokenizer,
            row["problem"],
            row["solution"],
            int(config["model"]["max_sequence_length"]),
        )
        for row in rows
    ]
    max_length = max(example["input_ids"].shape[1] for example in examples)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids, attention_masks, labels = [], [], []
    for example in examples:
        length = example["input_ids"].shape[1]
        padding = max_length - length
        input_ids.append(torch.nn.functional.pad(example["input_ids"], (0, padding), value=pad_id))
        attention_masks.append(
            torch.nn.functional.pad(example["attention_mask"], (0, padding), value=0)
        )
        labels.append(torch.nn.functional.pad(example["labels"], (0, padding), value=-100))
    return {
        "input_ids": torch.cat(input_ids).to(device),
        "attention_mask": torch.cat(attention_masks).to(device),
        "labels": torch.cat(labels).to(device),
        "truncated_count": sum(bool(example["truncated"]) for example in examples),
    }


def parameter_norms(model) -> dict:
    by_factor = {"A": [], "B": []}
    for name, parameter in trainable_lora_parameters(model):
        factor = "A" if "lora_A" in name else "B"
        by_factor[factor].append(parameter.detach().float().reshape(-1).cpu())
    return {
        factor: float(torch.linalg.vector_norm(torch.cat(parts)))
        for factor, parts in by_factor.items()
    }


def dtype_report(model) -> dict:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    return {
        "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable}),
        "frozen_dtypes": sorted({str(p.dtype) for _, p in frozen}),
        "trainable_parameter_count": sum(p.numel() for _, p in trainable),
        "frozen_parameter_count": sum(p.numel() for _, p in frozen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/gradient_geometry/qwen2.5_1.5b_math.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "result/MATH/Qwen2.5-1.5B-Instruct/gradient_geometry/lora_warmup_128_steps32",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolve_paths(config)
    seed = int(config["experiment"]["seed"])
    set_global_seed(seed)
    warmup = config["warmup"]
    started = time.time()

    math_train = load_math_train(Path(config["data"]["math_train_path"]))
    math500 = load_math500(Path(config["data"]["math500_path"]))
    fixed_splits = build_fixed_splits(
        math_train,
        math500,
        int(config["data"]["predictor_train_size"]),
        int(config["data"]["candidate_test_size"]),
        seed,
    )
    rows = select_warmup_rows(math_train, fixed_splits, int(warmup["size"]), seed)
    write_jsonl(args.output / "warmup_manifest.jsonl", [public_manifest_row(row, "lora_warmup") for row in rows])
    write_json(args.output / "config_resolved.json", config)
    write_json(
        args.output / "prompt.json",
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "assistant_target_field": "solution",
            "loss_masking": "prompt_tokens_are_-100; assistant_solution_tokens_only",
        },
    )

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    model.train()
    precision = dtype_report(model)
    if precision["trainable_dtypes"] != ["torch.float32"]:
        raise RuntimeError(f"LoRA parameters are not FP32: {precision}")
    before_norms = parameter_norms(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_lora_parameters(model)],
        lr=float(warmup["learning_rate"]),
        weight_decay=float(warmup["weight_decay"]),
    )
    batch_size = int(warmup["batch_size"])
    steps = int(warmup["steps"])
    if steps * batch_size != len(rows):
        raise ValueError("Warm-up currently requires steps * batch_size == size")
    losses = []
    gradient_norms = []
    truncated_total = 0
    for step in trange(steps, desc="LoRA warm-up"):
        batch = collate_batch(
            tokenizer,
            rows[step * batch_size : (step + 1) * batch_size],
            config,
            args.device,
        )
        truncated_total += batch.pop("truncated_count")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=bool(config["model"].get("forward_autocast_bfloat16", False)),
        ):
            output = model(**batch, return_dict=True)
        if not torch.isfinite(output.loss):
            raise FloatingPointError(f"Non-finite warm-up loss at step {step + 1}")
        backward_scale = float(config.get("loss", {}).get("backward_scale", 1.0))
        (output.loss * backward_scale).backward()
        if backward_scale != 1.0:
            for _, parameter in trainable_lora_parameters(model):
                if parameter.grad is not None:
                    parameter.grad.div_(backward_scale)
        named_parameters = trainable_lora_parameters(model)
        bad_gradients = [
            name for name, parameter in named_parameters
            if parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ]
        if bad_gradients:
            raise FloatingPointError(
                f"Non-finite LoRA gradients at step {step + 1}: {bad_gradients[:10]}"
            )
        # Accumulate the norm in FP64. With many FP32 LoRA tensors, the default
        # FP32 squared-norm reduction can overflow even when every element is finite.
        total_sq = torch.zeros((), dtype=torch.float64, device=args.device)
        for _, parameter in named_parameters:
            total_sq.add_(parameter.grad.double().square().sum())
        gradient_norm = total_sq.sqrt()
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite FP64 gradient norm at step {step + 1}")
        max_grad_norm = float(warmup.get("max_grad_norm", 1.0))
        clip_coefficient = min(1.0, max_grad_norm / (float(gradient_norm) + 1e-12))
        if clip_coefficient < 1.0:
            for _, parameter in named_parameters:
                parameter.grad.mul_(clip_coefficient)
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        optimizer.step()
        losses.append(float(output.loss.detach().float().cpu()))

    after_norms = parameter_norms(model)
    optimizer_state_dtypes = sorted(
        {
            str(value.dtype)
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
    )
    adapter_dir = args.output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "status": "complete",
        "formal_gradient_extraction_started": False,
        "warmup_samples": len(rows),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": float(warmup["learning_rate"]),
        "max_grad_norm": float(warmup.get("max_grad_norm", 1.0)),
        "backward_scale": float(config.get("loss", {}).get("backward_scale", 1.0)),
        "pre_clip_gradient_norms": gradient_norms,
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "truncated_samples": truncated_total,
        "parameter_norms_before": before_norms,
        "parameter_norms_after": after_norms,
        "precision": {
            **precision,
            "forward_autocast": (
                "torch.bfloat16"
                if bool(config["model"].get("forward_autocast_bfloat16", False))
                else "disabled"
            ),
            "optimizer_state_dtypes": optimizer_state_dtypes,
        },
        "adapter_path": str(adapter_dir.resolve()),
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
