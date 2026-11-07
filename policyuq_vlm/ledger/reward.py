from __future__ import annotations

import torch

from policyuq_vlm.basecamp.config import RewardConfig


def terminal_reward(
    pred_label: torch.Tensor,
    true_label: torch.Tensor,
    p_pred: torch.Tensor,
    p_ideal: torch.Tensor,
    lengths: torch.Tensor,
    cfg: RewardConfig,
) -> torch.Tensor:
    accuracy = (pred_label == true_label).to(p_pred.dtype)
    calibration = (p_pred - p_ideal).abs()
    steps = lengths.to(p_pred.dtype)
    return cfg.alpha * accuracy - cfg.beta * calibration - cfg.step_cost * steps
