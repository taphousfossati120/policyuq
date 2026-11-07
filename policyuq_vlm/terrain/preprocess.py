from __future__ import annotations

import torch
import torch.nn.functional as F


def resample_center_normalize(volume: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    resampled = F.interpolate(
        volume[None, None].float(),
        size=shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    mean = resampled.mean()
    std = resampled.std().clamp_min(1e-6)
    return (resampled - mean) / std


def crop_to_bbox(volume: torch.Tensor, bbox: torch.Tensor, margin: int = 4) -> torch.Tensor:
    z0, y0, x0, z1, y1, x1 = (int(v) for v in bbox.tolist())
    d, h, w = volume.shape
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(d, z1 + margin)
    y1 = min(h, y1 + margin)
    x1 = min(w, x1 + margin)
    return volume[z0:z1, y0:y1, x0:x1]
