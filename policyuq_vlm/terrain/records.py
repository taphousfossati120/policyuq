from __future__ import annotations

from typing import TypedDict

import torch
from attrs import frozen


@frozen
class MpMriCase:
    case_id: str
    t2w: torch.Tensor
    dwi: torch.Tensor
    adc: torch.Tensor
    bbox: torch.Tensor
    pirads: int
    label: int


class Batch(TypedDict):
    t2w: torch.Tensor
    dwi: torch.Tensor
    adc: torch.Tensor
    bbox: torch.Tensor
    pirads: torch.Tensor
    label: torch.Tensor


def collate(cases: list[MpMriCase]) -> Batch:
    return Batch(
        t2w=torch.stack([c.t2w for c in cases]),
        dwi=torch.stack([c.dwi for c in cases]),
        adc=torch.stack([c.adc for c in cases]),
        bbox=torch.stack([c.bbox for c in cases]),
        pirads=torch.tensor([c.pirads for c in cases], dtype=torch.long),
        label=torch.tensor([c.label for c in cases], dtype=torch.long),
    )


def sequences(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch["t2w"], batch["dwi"], batch["adc"]
