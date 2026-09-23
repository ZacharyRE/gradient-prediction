#!/usr/bin/env python3
"""Completion-only SFT data and optimization helpers; no adapters."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from itertools import islice
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gradient_geometry.extraction import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from gradient_geometry.sft_protocol import (artifact_identity, digest, exclusive_run,
    file_sha256, sample_identity, validate_manifest, versions, write_json)  # noqa: E402


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
        self.sample_ids = []
        lengths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            self.sample_ids.append(sample_identity(row))
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
                raise ValueError('Sequence exceeds model context; refusing truncation')
            input_ids = prompt + response
            labels = [-100] * len(prompt) + response
            if not response:
                raise RuntimeError(f"No supervised tokens for sample {row.get('sample_id')}")
            self.items.append({"input_ids": input_ids, "labels": labels})
            lengths.append(len(input_ids))
            source = row.get("source", "unknown")
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
        if not self.items:
            raise ValueError(f'Empty dataset: {path}')
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


class IndexedTrainingCollator:
    def __init__(self, dataset, collator):
        self.dataset, self.collator = dataset, collator

    def __call__(self, indices):
        return self.collator([self.dataset[i] for i in indices]), list(indices)


def autocast_context(device, precision='bfloat16'):
    if precision not in {'bfloat16', 'float32'}:
        raise ValueError(f'Unsupported precision: {precision}')
    return (torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=precision == 'bfloat16')
            if torch.device(device).type == 'cuda' else nullcontext())


def accumulate_gradients(model, window, device, normalization='token_mean', precision='bfloat16'):
    """Accumulate one update; callers clip/step once after this function."""
    if normalization not in {'token_mean', 'legacy_microbatch_mean'}:
        raise ValueError(f'Unknown loss normalization: {normalization}')
    counts = [int((batch['labels'][:, 1:] != -100).sum()) for batch in window]
    if not counts or min(counts) <= 0:
        raise ValueError('Every microbatch must have supervised causal tokens')
    total = sum(counts)
    token_loss_sum = objective_loss = 0.0
    for cpu_batch, count in zip(window, counts):
        batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
        with autocast_context(device, precision):
            loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        weight = count / total if normalization == 'token_mean' else 1 / len(window)
        (loss * weight).backward()
        token_loss_sum += float(loss.detach()) * count
        objective_loss += float(loss.detach()) * weight
    return dict(train_token_loss=token_loss_sum/total, train_objective_loss=objective_loss,
                supervised_tokens=total, microbatch_supervised_tokens=counts)


@torch.no_grad()
def evaluate(model, loader, device: str, precision='bfloat16') -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, precision):
            output = model(**batch, use_cache=False)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite development loss")
        count = int((batch["labels"][:, 1:] != -100).sum())
        loss_sum += float(output.loss) * count
        token_count += count
    model.train(was_training)
    return loss_sum / token_count

