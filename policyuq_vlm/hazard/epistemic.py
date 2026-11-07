from __future__ import annotations

import torch
from attrs import frozen


@frozen
class EpistemicEstimate:
    mean: torch.Tensor
    variance: torch.Tensor


def epistemic_variance(passes: torch.Tensor) -> EpistemicEstimate:
    mean = passes.mean(dim=0)
    variance = passes.var(dim=0, unbiased=False)
    return EpistemicEstimate(mean=mean, variance=variance)
