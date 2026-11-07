from __future__ import annotations

import torch

BENIGN = 0
SUSPICIOUS = 1
REFER = 2


def triage_decision(prob: torch.Tensor, total: torch.Tensor, delta: float) -> torch.Tensor:
    decision = torch.full_like(prob, BENIGN, dtype=torch.long)
    suspicious = (prob > 0.5) & (total <= delta)
    decision = torch.where(suspicious, torch.full_like(decision, SUSPICIOUS), decision)
    refer = total > delta
    return torch.where(refer, torch.full_like(decision, REFER), decision)


def _coverage(prob: torch.Tensor, total: torch.Tensor, label: torch.Tensor, delta: float) -> float:
    decision = triage_decision(prob, total, delta)
    positive = label == 1
    covered = positive & ((decision == SUSPICIOUS) | (decision == REFER))
    denom = int(positive.sum().item())
    if denom == 0:
        return 1.0
    return float(covered.sum().item()) / denom


def select_threshold(
    prob: torch.Tensor, total: torch.Tensor, label: torch.Tensor, sensitivity_target: float
) -> float:
    candidates = torch.unique(total).sort().values
    best = float(candidates[0].item())
    for value in candidates:
        delta = float(value.item())
        if _coverage(prob, total, label, delta) >= sensitivity_target:
            best = delta
    return best


def referral_rate(decision: torch.Tensor) -> float:
    return float((decision == REFER).float().mean().item())
