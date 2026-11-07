from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from attrs import field, frozen


@frozen
class VolumeConfig:
    height: int = 192
    width: int = 192
    depth: int = 48
    patch: tuple[int, int, int] = (4, 4, 2)
    window: tuple[int, int, int] = (6, 6, 6)
    embed_dim: int = 768
    stage_dims: tuple[int, ...] = (96, 192, 384, 768)
    in_channels: int = 1


@frozen
class EncoderConfig:
    volume: VolumeConfig = field(factory=VolumeConfig)
    fusion_layers: int = 4
    fusion_heads: int = 8
    vocab_size: int = 87
    temperature: float = 0.07
    text_dim: int = 768


@frozen
class PolicyConfig:
    gru_layers: int = 2
    gru_hidden: int = 512
    n_tokens: int = 432
    n_sequences: int = 3
    policy_hidden: int = 256
    max_steps: int = 12


@frozen
class UncertaintyConfig:
    mc_samples: int = 20
    stochastic_passes: int = 10
    dropout: float = 0.1
    sensitivity_target: float = 0.95


@frozen
class RewardConfig:
    alpha: float = 1.0
    beta: float = 0.5
    step_cost: float = 0.02


@frozen
class PPOConfig:
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy: float = 0.01
    epochs_per_round: int = 4


@frozen
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    batch_size: int = 16


@frozen
class PhaseConfig:
    pretrain_epochs: int = 50
    finetune_epochs: int = 30
    policy_epochs: int = 40
    policy_lr_scale: float = 0.1
    calibration_fraction: float = 0.15
    calibration_refresh: int = 5
    het_weight: float = 0.3
    early_stop_patience: int = 3


@frozen
class DataConfig:
    name: str = "pi_cai"
    root: str = "data/pi_cai"
    n_cases: int = 1500
    folds: int = 5
    seed: int = 0


@frozen
class AblationConfig:
    use_vl_alignment: bool = True
    use_policy: bool = True
    use_uncertainty: bool = True
    use_fusion: bool = True
    policy_algorithm: str = "ppo"


@frozen
class ExperimentConfig:
    name: str = "main"
    encoder: EncoderConfig = field(factory=EncoderConfig)
    policy: PolicyConfig = field(factory=PolicyConfig)
    uncertainty: UncertaintyConfig = field(factory=UncertaintyConfig)
    reward: RewardConfig = field(factory=RewardConfig)
    ppo: PPOConfig = field(factory=PPOConfig)
    optim: OptimConfig = field(factory=OptimConfig)
    phases: PhaseConfig = field(factory=PhaseConfig)
    data: DataConfig = field(factory=DataConfig)
    ablation: AblationConfig = field(factory=AblationConfig)
    world_size: int = 4
    precision: str = "amp"
    seed: int = 0


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _volume(raw: dict[str, Any]) -> VolumeConfig:
    spec = dict(raw)
    if "patch" in spec:
        spec["patch"] = _as_int_tuple(spec["patch"])
    if "window" in spec:
        spec["window"] = _as_int_tuple(spec["window"])
    if "stage_dims" in spec:
        spec["stage_dims"] = _as_int_tuple(spec["stage_dims"])
    return VolumeConfig(**spec)


def _encoder(raw: dict[str, Any]) -> EncoderConfig:
    spec = dict(raw)
    if "volume" in spec:
        spec["volume"] = _volume(spec["volume"])
    return EncoderConfig(**spec)


def experiment_from_mapping(raw: dict[str, Any]) -> ExperimentConfig:
    spec = dict(raw)
    if "encoder" in spec:
        spec["encoder"] = _encoder(spec["encoder"])
    if "policy" in spec:
        spec["policy"] = PolicyConfig(**spec["policy"])
    if "uncertainty" in spec:
        spec["uncertainty"] = UncertaintyConfig(**spec["uncertainty"])
    if "reward" in spec:
        spec["reward"] = RewardConfig(**spec["reward"])
    if "ppo" in spec:
        spec["ppo"] = PPOConfig(**spec["ppo"])
    if "optim" in spec:
        spec["optim"] = OptimConfig(**spec["optim"])
    if "phases" in spec:
        spec["phases"] = PhaseConfig(**spec["phases"])
    if "data" in spec:
        spec["data"] = DataConfig(**spec["data"])
    if "ablation" in spec:
        spec["ablation"] = AblationConfig(**spec["ablation"])
    return ExperimentConfig(**spec)


def _read_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_experiment(path: str | Path, base: str | Path | None = None) -> ExperimentConfig:
    raw = _read_toml(path)
    if base is not None:
        raw = _deep_merge(_read_toml(base), raw)
    return experiment_from_mapping(raw)
