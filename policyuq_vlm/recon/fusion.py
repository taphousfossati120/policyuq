from __future__ import annotations

from typing import cast

import torch
from torch import nn


class CrossSequenceFusion(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        b = evidence.shape[0]
        cls = self.cls.expand(b, -1, -1)
        sequence = torch.cat([cls, evidence], dim=1)
        fused = self.transformer(sequence)
        return cast(torch.Tensor, fused[:, 0])
