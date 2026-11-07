from __future__ import annotations

import torch

from policyuq_vlm.terrain.vocabulary import DESCRIPTORS

_T2_OFFSET = next(i for i, d in enumerate(DESCRIPTORS) if d.group == "signal_t2")


def descriptor_target(pirads: torch.Tensor) -> torch.Tensor:
    return _T2_OFFSET + (pirads.clamp(1, 5) - 1)
