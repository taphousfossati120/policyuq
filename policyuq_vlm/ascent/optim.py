from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from policyuq_vlm.basecamp.config import OptimConfig


def build_optimizer(
    params: Iterator[torch.nn.Parameter], cfg: OptimConfig, lr_scale: float = 1.0
) -> AdamW:
    return AdamW(params, lr=cfg.lr * lr_scale, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: AdamW, total_steps: int) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=max(1, total_steps))


def clip_gradients(params: Iterator[torch.nn.Parameter], max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(list(params), max_norm)
