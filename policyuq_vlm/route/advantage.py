from __future__ import annotations

import torch


def generalized_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(steps)):
        next_value = values[:, t + 1] if t + 1 < steps else torch.zeros_like(values[:, 0])
        next_mask = mask[:, t + 1] if t + 1 < steps else torch.zeros_like(mask[:, 0])
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last = delta + gamma * lam * next_mask * last
        advantages[:, t] = last * mask[:, t]
    returns = advantages + values
    return advantages, returns
