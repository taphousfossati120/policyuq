from __future__ import annotations

import torch
from attrs import frozen


@frozen
class EvidenceLibrary:
    evidence: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.evidence.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.evidence.shape[1])

    @property
    def commit_index(self) -> int:
        return self.n_tokens

    @property
    def n_actions(self) -> int:
        return self.n_tokens + 1

    def feature(self, action: torch.Tensor) -> torch.Tensor:
        clamped = action.clamp(max=self.n_tokens - 1)
        index = clamped.view(-1, 1, 1).expand(-1, 1, self.evidence.shape[-1])
        return self.evidence.gather(1, index).squeeze(1)
