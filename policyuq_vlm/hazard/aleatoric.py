from __future__ import annotations

import torch
from torch import nn


class HeteroscedasticHead(nn.Module):
    def __init__(self, state_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(state_dim, num_classes)
        self.log_variance = nn.Linear(state_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(state)
        log_var = self.log_variance(state).squeeze(-1)
        return logits, log_var


def heteroscedastic_loss(
    logits: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    sigma = torch.exp(0.5 * log_var)
    noise = torch.randn(samples, *logits.shape, device=logits.device, dtype=logits.dtype)
    perturbed = logits.unsqueeze(0) + sigma.view(1, -1, 1) * noise
    log_prob = torch.log_softmax(perturbed, dim=-1)
    index = target.view(1, -1, 1).expand(samples, -1, 1)
    picked = log_prob.gather(-1, index).squeeze(-1)
    return -picked.mean()


def aleatoric_variance(log_var: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_var)
