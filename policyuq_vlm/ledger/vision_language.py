from __future__ import annotations

import torch
import torch.nn.functional as F


def vision_language_loss(
    aligned: torch.Tensor,
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    sequences = aligned.shape[1]
    total = aligned.new_zeros(())
    for s in range(sequences):
        logits = aligned[:, s] @ descriptors.t() / temperature
        total = total + F.cross_entropy(logits, targets)
    return total / sequences
