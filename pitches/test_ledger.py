from __future__ import annotations

import torch

from policyuq_vlm.basecamp.config import RewardConfig
from policyuq_vlm.ledger.reward import terminal_reward
from policyuq_vlm.ledger.vision_language import vision_language_loss


def _reward(correct: bool, calib: float, steps: int) -> float:
    pred = torch.tensor([1])
    true = torch.tensor([1 if correct else 0])
    p_pred = torch.tensor([0.5 + calib])
    p_ideal = torch.tensor([0.5])
    lengths = torch.tensor([steps])
    value = terminal_reward(pred, true, p_pred, p_ideal, lengths, RewardConfig())
    return float(value[0])


def test_reward_rewards_accuracy() -> None:
    assert _reward(True, 0.0, 1) > _reward(False, 0.0, 1)


def test_reward_penalizes_miscalibration() -> None:
    assert _reward(True, 0.0, 1) > _reward(True, 0.3, 1)


def test_reward_penalizes_longer_trajectories() -> None:
    assert _reward(True, 0.0, 1) > _reward(True, 0.0, 6)


def test_vision_language_loss_low_when_aligned() -> None:
    descriptors = torch.eye(4)
    aligned = torch.stack([torch.eye(4)[[0, 1]]], dim=1)
    targets = torch.tensor([0, 1])
    matched = vision_language_loss(aligned, descriptors, targets, temperature=0.07)
    mismatched = vision_language_loss(aligned, descriptors, torch.tensor([2, 3]), temperature=0.07)
    assert matched < mismatched
