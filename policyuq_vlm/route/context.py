from __future__ import annotations

import torch
from torch import nn


class ContextGRU(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.cells = nn.ModuleList(
            [nn.GRUCell(input_dim if i == 0 else hidden, hidden) for i in range(layers)]
        )

    def initial(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        return [torch.zeros(batch_size, self.hidden, device=device) for _ in self.cells]

    def step(self, hiddens: list[torch.Tensor], token: torch.Tensor) -> list[torch.Tensor]:
        updated: list[torch.Tensor] = []
        signal = token
        for cell, state in zip(self.cells, hiddens, strict=True):
            signal = cell(signal, state)
            updated.append(signal)
        return updated

    @staticmethod
    def readout(hiddens: list[torch.Tensor]) -> torch.Tensor:
        return hiddens[-1]
