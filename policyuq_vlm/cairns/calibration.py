from __future__ import annotations

import torch


def expected_calibration_error(prob: torch.Tensor, label: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=prob.device)
    target = label.to(prob.dtype)
    total = prob.shape[0]
    error = prob.new_zeros(())
    for i in range(bins):
        lower = edges[i]
        upper = edges[i + 1]
        in_bin = (prob > lower) & (prob <= upper) if i > 0 else (prob >= lower) & (prob <= upper)
        count = in_bin.sum()
        if int(count.item()) == 0:
            continue
        confidence = prob[in_bin].mean()
        accuracy = target[in_bin].mean()
        error = error + (count / total) * (confidence - accuracy).abs()
    return float(error.item())


def brier_score(prob: torch.Tensor, label: torch.Tensor) -> float:
    target = label.to(prob.dtype)
    return float((prob - target).pow(2).mean().item())


def reliability_bins(
    prob: torch.Tensor, label: torch.Tensor, bins: int = 15
) -> tuple[list[float], list[float]]:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=prob.device)
    target = label.to(prob.dtype)
    confidences: list[float] = []
    accuracies: list[float] = []
    for i in range(bins):
        in_bin = (prob > edges[i]) & (prob <= edges[i + 1])
        if int(in_bin.sum().item()) == 0:
            continue
        confidences.append(float(prob[in_bin].mean().item()))
        accuracies.append(float(target[in_bin].mean().item()))
    return confidences, accuracies
