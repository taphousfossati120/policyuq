from __future__ import annotations

from typing import cast

import torch
from torch import nn

from policyuq_vlm.basecamp.config import EncoderConfig

DOWNSAMPLE = 16


def _grid(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        max(1, shape[0] // DOWNSAMPLE),
        max(1, shape[1] // DOWNSAMPLE),
        max(1, shape[2] // DOWNSAMPLE),
    )


def _divisor(requested: int, extent: int) -> int:
    size = min(requested, extent)
    while extent % size != 0:
        size -= 1
    return size


def window_partition(x: torch.Tensor, window: tuple[int, int, int]) -> torch.Tensor:
    b, d, h, w, c = x.shape
    wd, wh, ww = window
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, c)


def window_reverse(
    windows: torch.Tensor, window: tuple[int, int, int], dims: tuple[int, int, int]
) -> torch.Tensor:
    d, h, w = dims
    wd, wh, ww = window
    c = windows.shape[-1]
    x = windows.view(-1, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(-1, d, h, w, c)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window: tuple[int, int, int], heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        size = window[0] * window[1] * window[2]
        self.bias = nn.Parameter(torch.zeros(heads, size, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.bias
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return cast(torch.Tensor, self.proj(out))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window: tuple[int, int, int],
        heads: int,
        shift: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, dims: tuple[int, int, int]) -> torch.Tensor:
        b, n, c = x.shape
        d, h, w = dims
        shortcut = x
        x = self.norm1(x).view(b, d, h, w, c)
        if any(self.shift):
            x = torch.roll(
                x, shifts=(-self.shift[0], -self.shift[1], -self.shift[2]), dims=(1, 2, 3)
            )
        windows = window_partition(x, self.window)
        attended = self.attn(windows)
        x = window_reverse(attended, self.window, dims)
        if any(self.shift):
            x = torch.roll(x, shifts=self.shift, dims=(1, 2, 3))
        x = x.view(b, n, c)
        x = shortcut + x
        return cast(torch.Tensor, x + self.mlp(self.norm2(x)))


class SwinStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        window: tuple[int, int, int],
        heads: int,
        depth: int,
    ) -> None:
        super().__init__()
        shift = (window[0] // 2, window[1] // 2, window[2] // 2)
        self.blocks = nn.ModuleList(
            [
                SwinBlock(in_dim, window, heads, (0, 0, 0) if i % 2 == 0 else shift)
                for i in range(depth)
            ]
        )
        self.project = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, dims: tuple[int, int, int]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, dims)
        return cast(torch.Tensor, self.project(x))


class Swin3DEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.grid = _grid((cfg.volume.depth, cfg.volume.height, cfg.volume.width))
        self.window = (
            _divisor(cfg.volume.window[2], self.grid[0]),
            _divisor(cfg.volume.window[0], self.grid[1]),
            _divisor(cfg.volume.window[1], self.grid[2]),
        )
        dims = cfg.volume.stage_dims
        self.embed = nn.Conv3d(
            cfg.volume.in_channels, dims[0], kernel_size=DOWNSAMPLE, stride=DOWNSAMPLE
        )
        self.embed_norm = nn.LayerNorm(dims[0])
        widths = [dims[0], *dims]
        self.stages = nn.ModuleList(
            [
                SwinStage(widths[i], widths[i + 1], self.window, cfg.fusion_heads, depth=2)
                for i in range(len(dims))
            ]
        )
        self.out_dim = dims[-1]

    @property
    def n_tokens(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x = self.embed(volume)
        _, _, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.embed_norm(tokens)
        dims = (d, h, w)
        for stage in self.stages:
            tokens = stage(tokens, dims)
        return cast(torch.Tensor, tokens)
