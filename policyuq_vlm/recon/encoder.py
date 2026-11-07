from __future__ import annotations

import torch
from attrs import frozen
from torch import nn

from policyuq_vlm.basecamp.config import EncoderConfig
from policyuq_vlm.recon.alignment import DescriptorAligner
from policyuq_vlm.recon.fusion import CrossSequenceFusion
from policyuq_vlm.recon.swin3d import Swin3DEncoder
from policyuq_vlm.recon.text_encoder import FrozenDescriptorEncoder
from policyuq_vlm.terrain.records import Batch, sequences
from policyuq_vlm.terrain.vocabulary import DESCRIPTORS


@frozen
class EncoderOutput:
    evidence: torch.Tensor
    fused: torch.Tensor
    aligned: torch.Tensor


class VisionLanguageEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, use_fusion: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_fusion = use_fusion
        self.backbone = Swin3DEncoder(cfg)
        dim = self.backbone.out_dim
        self.aligner = DescriptorAligner(dim)
        self.text = FrozenDescriptorEncoder(DESCRIPTORS, cfg.text_dim)
        self.text_project = nn.Linear(cfg.text_dim, dim)
        self.fusion = CrossSequenceFusion(dim, cfg.fusion_layers, cfg.fusion_heads)

    @property
    def n_tokens(self) -> int:
        return self.backbone.n_tokens

    @property
    def dim(self) -> int:
        return self.backbone.out_dim

    def descriptor_embeddings(self) -> torch.Tensor:
        table = self.text_project(self.text())
        return nn.functional.normalize(table, dim=-1)

    def forward(self, batch: Batch) -> EncoderOutput:
        token_maps = [self.backbone(volume) for volume in sequences(batch)]
        evidence = torch.cat(token_maps, dim=1)
        aligned = torch.stack([self.aligner(tokens) for tokens in token_maps], dim=1)
        if self.use_fusion:
            fused = self.fusion(evidence)
        else:
            fused = evidence.mean(dim=1)
        return EncoderOutput(evidence=evidence, fused=fused, aligned=aligned)
