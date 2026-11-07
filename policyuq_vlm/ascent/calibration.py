from __future__ import annotations

import torch


class CalibrationTarget:
    def __init__(self, bins: int = 10) -> None:
        self.bins = bins
        self._accuracy = torch.full((bins,), 0.5)

    def update(self, confidence: torch.Tensor, correct: torch.Tensor) -> None:
        edges = torch.linspace(0.0, 1.0, self.bins + 1)
        updated = self._accuracy.clone()
        for i in range(self.bins):
            in_bin = (confidence > edges[i]) & (confidence <= edges[i + 1])
            if int(in_bin.sum().item()) > 0:
                updated[i] = correct[in_bin].float().mean()
        self._accuracy = updated

    def lookup(self, confidence: torch.Tensor) -> torch.Tensor:
        index = (confidence * self.bins).long().clamp(0, self.bins - 1)
        return self._accuracy.to(confidence.device)[index]

    def state(self) -> torch.Tensor:
        return self._accuracy.clone()
