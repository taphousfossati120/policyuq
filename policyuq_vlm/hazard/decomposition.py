from __future__ import annotations

import torch
from attrs import frozen


@frozen
class UncertaintyBudget:
    aleatoric: torch.Tensor
    epistemic: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.aleatoric + self.epistemic


def decompose(aleatoric: torch.Tensor, epistemic: torch.Tensor) -> UncertaintyBudget:
    return UncertaintyBudget(aleatoric=aleatoric, epistemic=epistemic)
