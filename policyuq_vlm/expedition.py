from __future__ import annotations

from policyuq_vlm.basecamp.config import ExperimentConfig
from policyuq_vlm.basecamp.seeding import set_seed
from policyuq_vlm.model import PolicyUQVLM
from policyuq_vlm.terrain.augment import AugmentConfig
from policyuq_vlm.terrain.datasets import MpMriDataset, build_dataset


def volume_shape(cfg: ExperimentConfig) -> tuple[int, int, int]:
    volume = cfg.encoder.volume
    return (volume.depth, volume.height, volume.width)


def build_model(cfg: ExperimentConfig) -> PolicyUQVLM:
    set_seed(cfg.seed)
    return PolicyUQVLM(cfg)


def build_splits(cfg: ExperimentConfig) -> tuple[MpMriDataset, MpMriDataset]:
    shape = volume_shape(cfg)
    full = build_dataset(cfg.data, shape, augment=AugmentConfig())
    cases = full.cases
    n_val = max(1, len(cases) // 5)
    val_cases = cases[:n_val]
    train_cases = cases[n_val:] or cases
    train = MpMriDataset(train_cases, shape, augment=AugmentConfig(), seed=cfg.data.seed)
    val = MpMriDataset(val_cases, shape, augment=None, seed=cfg.data.seed + 1)
    return train, val
