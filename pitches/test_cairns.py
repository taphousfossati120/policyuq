from __future__ import annotations

import torch

from policyuq_vlm.cairns.calibration import brier_score, expected_calibration_error
from policyuq_vlm.cairns.discrimination import auroc, sensitivity_at_specificity
from policyuq_vlm.cairns.statistics import bootstrap_ci


def test_auroc_known_value() -> None:
    prob = torch.tensor([0.1, 0.4, 0.35, 0.8])
    label = torch.tensor([0, 0, 1, 1])
    assert abs(auroc(prob, label) - 0.75) < 1e-6


def test_brier_perfect_and_worst() -> None:
    label = torch.tensor([1.0, 0.0])
    assert brier_score(torch.tensor([1.0, 0.0]), label) == 0.0
    assert abs(brier_score(torch.tensor([0.5, 0.5]), label) - 0.25) < 1e-6


def test_sensitivity_separable() -> None:
    prob = torch.tensor([0.05, 0.1, 0.9, 0.95])
    label = torch.tensor([0, 0, 1, 1])
    assert sensitivity_at_specificity(prob, label, 0.9) == 1.0


def test_ece_increases_with_overconfidence() -> None:
    label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    confident_wrong = torch.tensor([0.05, 0.95, 0.05, 0.95])
    calibrated = torch.tensor([0.5, 0.5, 0.5, 0.5])
    assert expected_calibration_error(confident_wrong, label) > expected_calibration_error(
        calibrated, label
    )


def test_bootstrap_ci_brackets_point() -> None:
    torch.manual_seed(0)
    prob = torch.rand(64)
    label = (prob > 0.5).long()
    low, high = bootstrap_ci(prob, label, n_boot=200)
    assert 0.0 <= low <= high <= 1.0
