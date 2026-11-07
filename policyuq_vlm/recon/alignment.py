from __future__ import annotations

import torch
from torch import nn


class DescriptorAligner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        projected = self.project(pooled)
        return nn.functional.normalize(projected, dim=-1)
