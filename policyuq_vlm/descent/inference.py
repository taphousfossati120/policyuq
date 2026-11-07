from __future__ import annotations

from typing import cast

import torch
from attrs import frozen
from torch.utils.data import DataLoader

from policyuq_vlm.model import PolicyUQVLM
from policyuq_vlm.terrain.datasets import MpMriDataset
from policyuq_vlm.terrain.records import Batch, MpMriCase, collate


@frozen
class Predictions:
    prob: torch.Tensor
    aleatoric: torch.Tensor
    epistemic: torch.Tensor
    label: torch.Tensor
    pirads: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.aleatoric + self.epistemic


def _move(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        t2w=batch["t2w"].to(device),
        dwi=batch["dwi"].to(device),
        adc=batch["adc"].to(device),
        bbox=batch["bbox"].to(device),
        pirads=batch["pirads"].to(device),
        label=batch["label"].to(device),
    )


def collect_predictions(
    model: PolicyUQVLM,
    dataset: MpMriDataset,
    device: torch.device,
    passes: int,
    batch_size: int = 16,
) -> Predictions:
    loader: DataLoader[MpMriCase] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    probs: list[torch.Tensor] = []
    ales: list[torch.Tensor] = []
    epis: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    pirads: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move(cast(Batch, batch), device)
            prob, ale, epi = model.uncertainty(moved, passes)
            probs.append(prob.cpu())
            ales.append(ale.cpu())
            epis.append(epi.cpu())
            labels.append(moved["label"].cpu())
            pirads.append(moved["pirads"].cpu())
    model.train()
    return Predictions(
        prob=torch.cat(probs),
        aleatoric=torch.cat(ales),
        epistemic=torch.cat(epis),
        label=torch.cat(labels),
        pirads=torch.cat(pirads),
    )
