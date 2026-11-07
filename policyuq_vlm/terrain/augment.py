from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from attrs import frozen


@frozen
class AugmentConfig:
    flip_prob: float = 0.5
    max_rotation_deg: float = 15.0
    elastic_separation: int = 64
    elastic_magnitude: float = 10.0
    gaussian_sigma: float = 0.05


def _rotation_grid(
    shape: tuple[int, int, int], angle_deg: float, dtype: torch.dtype
) -> torch.Tensor:
    angle = math.radians(angle_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    theta = torch.tensor(
        [[[cos, -sin, 0.0, 0.0], [sin, cos, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=dtype
    )
    return F.affine_grid(theta, [1, 1, *shape], align_corners=False)


def _elastic_offsets(
    shape: tuple[int, int, int],
    cfg: AugmentConfig,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    coarse = [max(2, dim // cfg.elastic_separation) for dim in shape]
    field = torch.randn(1, 3, *coarse, generator=generator, dtype=dtype)
    field = F.interpolate(field, size=shape, mode="trilinear", align_corners=False)
    scale = cfg.elastic_magnitude / max(shape)
    return field.permute(0, 2, 3, 4, 1) * scale


def apply_augmentations(
    volume: torch.Tensor, cfg: AugmentConfig, generator: torch.Generator
) -> torch.Tensor:
    out = volume
    if torch.rand(1, generator=generator).item() < cfg.flip_prob:
        out = torch.flip(out, dims=(2,))
    shape = (out.shape[0], out.shape[1], out.shape[2])
    angle = (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * cfg.max_rotation_deg
    grid = _rotation_grid(shape, angle, out.dtype) + _elastic_offsets(
        shape, cfg, generator, out.dtype
    )
    warped = F.grid_sample(out[None, None], grid, align_corners=False, mode="bilinear")[0, 0]
    noise = torch.randn(warped.shape, generator=generator, dtype=warped.dtype) * cfg.gaussian_sigma
    return warped + noise
