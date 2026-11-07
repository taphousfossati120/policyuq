from __future__ import annotations

import torch
from attrs import frozen


@frozen
class PolicyLosses:
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _normalize(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    centered = advantages - _masked_mean(advantages, mask)
    variance = _masked_mean(centered.pow(2), mask)
    return centered / variance.clamp_min(1e-8).sqrt()


def reinforce_loss(
    new_logp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    normed = _normalize(advantages, mask)
    return -_masked_mean(new_logp * normed, mask)


def dpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    preference = torch.sign(_normalize(advantages, mask))
    logratio = new_logp - old_logp
    return -_masked_mean(torch.nn.functional.logsigmoid(beta * preference * logratio), mask)


def ppo_surrogate(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    entropy: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
) -> PolicyLosses:
    normed = advantages - _masked_mean(advantages, mask)
    variance = _masked_mean(normed.pow(2), mask)
    normed = normed / variance.clamp_min(1e-8).sqrt()
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * normed
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * normed
    policy = -_masked_mean(torch.minimum(unclipped, clipped), mask)
    value = _masked_mean((returns - values).pow(2), mask)
    entropy_term = _masked_mean(entropy, mask)
    return PolicyLosses(policy=policy, value=value, entropy=entropy_term)
