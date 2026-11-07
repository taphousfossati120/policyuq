from __future__ import annotations

import torch

from policyuq_vlm.basecamp.config import EncoderConfig, VolumeConfig
from policyuq_vlm.recon.encoder import VisionLanguageEncoder
from policyuq_vlm.recon.swin3d import Swin3DEncoder


def _encoder_config() -> EncoderConfig:
    return EncoderConfig(
        volume=VolumeConfig(height=32, width=32, depth=16, stage_dims=(8, 8, 8, 16)),
        fusion_layers=1,
        fusion_heads=2,
        text_dim=16,
    )


def test_swin_token_grid() -> None:
    backbone = Swin3DEncoder(_encoder_config())
    out = backbone(torch.randn(2, 1, 16, 32, 32))
    assert out.shape == (2, backbone.n_tokens, backbone.out_dim)
    assert backbone.n_tokens == 1 * 2 * 2


def test_paper_token_count() -> None:
    cfg = EncoderConfig(volume=VolumeConfig())
    backbone = Swin3DEncoder(cfg)
    assert backbone.n_tokens == 12 * 12 * 3
    assert backbone.n_tokens == 432


def test_backbone_weights_shared() -> None:
    encoder = VisionLanguageEncoder(_encoder_config())
    backbones = [m for m in encoder.modules() if isinstance(m, Swin3DEncoder)]
    assert len(backbones) == 1
