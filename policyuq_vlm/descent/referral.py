from __future__ import annotations

import torch

from policyuq_vlm.descent.inference import Predictions
from policyuq_vlm.hazard.triage import (
    REFER,
    SUSPICIOUS,
    select_threshold,
    triage_decision,
)


def _sensitivity(prob: torch.Tensor, decision: torch.Tensor, label: torch.Tensor) -> float:
    positive = label == 1
    denom = int(positive.sum().item())
    if denom == 0:
        return 1.0
    covered = positive & ((decision == SUSPICIOUS) | (decision == REFER))
    return float(covered.sum().item()) / denom


def _false_positive_reduction(
    prob: torch.Tensor, decision: torch.Tensor, label: torch.Tensor
) -> float:
    negative = label == 0
    baseline = int((negative & (prob > 0.5)).sum().item())
    if baseline == 0:
        return 0.0
    referred = negative & (decision == REFER) & (prob > 0.5)
    return float(referred.sum().item()) / baseline


def _analyze(
    prob: torch.Tensor, total: torch.Tensor, label: torch.Tensor, target: float
) -> dict[str, float]:
    delta = select_threshold(prob, total, label, target)
    decision = triage_decision(prob, total, delta)
    return {
        "threshold": delta,
        "referral_rate": float((decision == REFER).float().mean().item()),
        "sensitivity": _sensitivity(prob, decision, label),
        "fp_reduction": _false_positive_reduction(prob, decision, label),
    }


def referral_analysis(
    predictions: Predictions, target: float = 0.95
) -> dict[str, dict[str, float]]:
    return {
        "combined": _analyze(predictions.prob, predictions.total, predictions.label, target),
        "aleatoric_only": _analyze(
            predictions.prob, predictions.aleatoric, predictions.label, target
        ),
        "epistemic_only": _analyze(
            predictions.prob, predictions.epistemic, predictions.label, target
        ),
    }
