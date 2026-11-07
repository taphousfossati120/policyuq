from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from policyuq_vlm.basecamp.config import DataConfig
from policyuq_vlm.terrain.augment import AugmentConfig, apply_augmentations
from policyuq_vlm.terrain.preprocess import resample_center_normalize
from policyuq_vlm.terrain.records import MpMriCase

_SUPPORTED = {"pi_cai", "prostatex", "prostate158"}


class MpMriDataset(Dataset[MpMriCase]):
    def __init__(
        self,
        cases: list[MpMriCase],
        shape: tuple[int, int, int],
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self._cases = cases
        self._shape = shape
        self._augment = augment
        self._seed = seed

    @property
    def cases(self) -> list[MpMriCase]:
        return self._cases

    def __len__(self) -> int:
        return len(self._cases)

    def _prepare(self, volume: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        prepared = resample_center_normalize(volume[0], self._shape)
        if self._augment is not None:
            prepared = apply_augmentations(prepared, self._augment, gen)
        return prepared[None]

    def __getitem__(self, index: int) -> MpMriCase:
        case = self._cases[index]
        gen = torch.Generator()
        gen.manual_seed(self._seed * 100003 + index)
        return MpMriCase(
            case_id=case.case_id,
            t2w=self._prepare(case.t2w, gen),
            dwi=self._prepare(case.dwi, gen),
            adc=self._prepare(case.adc, gen),
            bbox=case.bbox,
            pirads=case.pirads,
            label=case.label,
        )


def _discover_cases(cfg: DataConfig) -> list[MpMriCase]:
    root = Path(cfg.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    raise NotImplementedError(
        f"A dataset-specific reader for '{cfg.name}' under {root} has not been implemented."
    )


def build_dataset(
    cfg: DataConfig,
    shape: tuple[int, int, int],
    augment: AugmentConfig | None = None,
) -> MpMriDataset:
    if cfg.name not in _SUPPORTED:
        raise ValueError(f"unknown dataset '{cfg.name}', expected one of {sorted(_SUPPORTED)}")
    cases = _discover_cases(cfg)
    return MpMriDataset(cases, shape, augment=augment, seed=cfg.seed)
