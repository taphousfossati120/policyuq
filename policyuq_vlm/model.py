from __future__ import annotations

from typing import cast

import torch
from attrs import frozen
from torch import nn

from policyuq_vlm.basecamp.config import ExperimentConfig
from policyuq_vlm.hazard.aleatoric import HeteroscedasticHead, aleatoric_variance
from policyuq_vlm.hazard.epistemic import epistemic_variance
from policyuq_vlm.recon.encoder import EncoderOutput, VisionLanguageEncoder
from policyuq_vlm.route.reasoner import Reasoner, Trajectory
from policyuq_vlm.route.state import EvidenceLibrary
from policyuq_vlm.terrain.records import Batch


@frozen
class ModelOutput:
    logits: torch.Tensor
    prob: torch.Tensor
    log_var: torch.Tensor
    aligned: torch.Tensor
    trajectory: Trajectory
    library: EvidenceLibrary
    final_state: torch.Tensor


class PolicyUQVLM(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_policy = cfg.ablation.use_policy
        self.encoder = VisionLanguageEncoder(cfg.encoder, use_fusion=cfg.ablation.use_fusion)
        self.reasoner = Reasoner(cfg.policy, self.encoder.dim, self.encoder.dim)
        self.head = HeteroscedasticHead(self.reasoner.state_dim)

    def encode(self, batch: Batch) -> EncoderOutput:
        return cast(EncoderOutput, self.encoder(batch))

    def _degenerate_state(self, fused: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(
            fused.shape[0], self.reasoner.state_dim - fused.shape[1], device=fused.device
        )
        return torch.cat([fused, pad], dim=-1)

    def _degenerate_trajectory(self, fused: torch.Tensor) -> Trajectory:
        batch = fused.shape[0]
        device = fused.device
        state = self._degenerate_state(fused).unsqueeze(1)
        zeros = torch.zeros(batch, 1, device=device)
        return Trajectory(
            states=state,
            actions=torch.zeros(batch, 1, dtype=torch.long, device=device),
            logprobs=zeros,
            values=zeros,
            entropies=zeros,
            mask=zeros,
            lengths=torch.ones(batch, dtype=torch.long, device=device),
            final_state=state.squeeze(1),
        )

    def forward(self, batch: Batch, stochastic: bool = False) -> ModelOutput:
        encoded = self.encode(batch)
        library = EvidenceLibrary(encoded.evidence)
        if self.use_policy:
            trajectory = self.reasoner.rollout(library, encoded.fused, stochastic)
        else:
            trajectory = self._degenerate_trajectory(encoded.fused)
        logits, log_var = self.head(trajectory.final_state)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        return ModelOutput(
            logits=logits,
            prob=prob,
            log_var=log_var,
            aligned=encoded.aligned,
            trajectory=trajectory,
            library=library,
            final_state=trajectory.final_state,
        )

    def classify_fused(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch)
        state = self._degenerate_state(encoded.fused)
        logits, log_var = self.head(state)
        return logits, log_var

    def descriptor_alignment(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch)
        return encoded.aligned, self.encoder.descriptor_embeddings()

    def epistemic_passes(self, batch: Batch, passes: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for _ in range(passes):
            result = self.forward(batch, stochastic=True)
            outputs.append(result.prob)
        return torch.stack(outputs, dim=0)

    def uncertainty(
        self, batch: Batch, passes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deterministic = self.forward(batch, stochastic=False)
        stochastic = self.epistemic_passes(batch, passes)
        epi = epistemic_variance(stochastic).variance
        ale = aleatoric_variance(deterministic.log_var)
        return deterministic.prob, ale, epi
