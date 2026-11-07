from __future__ import annotations

import math

import torch
from torch import nn

from policyuq_vlm.route.state import EvidenceLibrary


class PolicyAgent(nn.Module):
    def __init__(self, state_dim: int, token_dim: int, hidden: int) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.query_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )
        self.commit_head = nn.Linear(state_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self, state_vec: torch.Tensor, library: EvidenceLibrary
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_head(state_vec)
        token_logits = torch.einsum("bnd,bd->bn", library.evidence, query) / math.sqrt(
            self.token_dim
        )
        commit = self.commit_head(state_vec)
        logits = torch.cat([token_logits, commit], dim=-1)
        value = self.value_head(state_vec).squeeze(-1)
        return logits, value
