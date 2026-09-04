#!/usr/bin/env python3
"""Minimal two-GPU NCCL diagnostic for the GRPO trainer/server pair."""

import os

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
x = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
dist.all_reduce(x)
print(
    f"rank={rank} local_rank={os.environ.get('LOCAL_RANK')} "
    f"uuid={torch.cuda.get_device_properties(rank).uuid} reduced={x.item()}",
    flush=True,
)
dist.destroy_process_group()
