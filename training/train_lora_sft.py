#!/usr/bin/env python3
"""Train a completion-only LoRA SFT adapter on a JSONL problem/solution set."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradient_geometry.extraction import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def chat_prompt_ids(tokenizer, problem: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(problem=problem)},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


class CompletionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.items = []
        self.source_counts: dict[str, int] = {}
        self.truncated = 0
        lengths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            prompt = chat_prompt_ids(tokenizer, row["problem"])
            response = tokenizer(row["solution"], add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                response = response + [tokenizer.eos_token_id]
            available = max_length - len(prompt)
            if available <= 0:
                raise RuntimeError(
                    f"Prompt alone exceeds max_length for sample {row.get('sample_id')}"
                )
            if len(response) > available:
                response = response[:available]
                self.truncated += 1
            input_ids = prompt + response
            labels = [-100] * len(prompt) + response
            if not response:
                raise RuntimeError(f"No supervised tokens for sample {row.get('sample_id')}")
            self.items.append({"input_ids": input_ids, "labels": labels})
            lengths.append(len(input_ids))
            source = row.get("source", "unknown")
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
        self.length_statistics = {
            "min": min(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": max(lengths),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Collator:
    def __init__(self, pad_token_id: int, multiple: int = 8):
        self.pad_token_id = pad_token_id
        self.multiple = multiple

    def __call__(self, items):
        longest = max(len(item["input_ids"]) for item in items)
        length = int(math.ceil(longest / self.multiple) * self.multiple)
        input_ids, labels, masks = [], [], []
        for item in items:
            padding = length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            masks.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch, use_cache=False)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite development loss")
        count = int((batch["labels"] != -100).sum())
        loss_sum += float(output.loss) * count
        token_count += count
    model.train()
    return loss_sum / token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["all-linear"])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        existing = json.loads((args.output / "summary.json").read_text())
        if existing.get("status") == "passed":
            print(json.dumps(existing, indent=2))
            return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = CompletionDataset(args.train_file, tokenizer, args.max_length)
    dev_data = CompletionDataset(args.dev_file, tokenizer, args.max_length)
    collator = Collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
        generator=generator, pin_memory=True, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
        pin_memory=True, num_workers=0,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    dataset_summary = {
        "train_file": str(args.train_file.resolve()),
        "dev_file": str(args.dev_file.resolve()),
        "train_samples": len(train_data),
        "dev_samples": len(dev_data),
        "train_source_counts": train_data.source_counts,
        "dev_source_counts": dev_data.source_counts,
        "train_truncated_samples": train_data.truncated,
        "dev_truncated_samples": dev_data.truncated,
        "train_length_statistics": train_data.length_statistics,
        "dev_length_statistics": dev_data.length_statistics,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
    }
    write_json(args.output / "dataset_summary.json", dataset_summary)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        # SDPA backward can produce NaNs for some right-padded BF16 batches in
        # the installed Torch/Transformers combination. Eager attention is
        # numerically stable for this model and preserves the requested BF16.
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            target_modules=args.target_modules[0] if args.target_modules == ["all-linear"] else args.target_modules,
        ),
    )
    model.to(args.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay, fused=True
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    optimizer.zero_grad(set_to_none=True)
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        accumulated_loss = 0.0
        accumulated_batches = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"SFT epoch {epoch}")):
            batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_index}"
                )
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach())
            accumulated_batches += 1
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, batch={batch_index}"
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    history.append({
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss_recent": accumulated_loss / accumulated_batches,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gradient_norm": float(grad_norm),
                    })
                    accumulated_loss = 0.0
                    accumulated_batches = 0
        dev_loss = evaluate(model, dev_loader, args.device)
        checkpoint = args.output / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        history.append({"epoch": epoch, "global_step": global_step, "dev_loss": dev_loss})
        write_json(args.output / "history.json", history)

    final = args.output / "final_adapter"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    summary = {
        "status": "passed",
        "model": str(args.model.resolve()),
        "precision": "bfloat16",
        "objective": "completion_only_supervised_cross_entropy",
        "lora": {
            "target_modules": args.target_modules,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "trainable_parameters": trainable_count,
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_steps,
            "max_grad_norm": args.max_grad_norm,
        },
        "dataset": dataset_summary,
        "epoch_dev_losses": [row for row in history if "dev_loss" in row],
        "elapsed_seconds": time.time() - started,
        "final_adapter": str(final),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
