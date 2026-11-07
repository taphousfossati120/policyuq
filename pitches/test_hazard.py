from __future__ import annotations

import torch

from policyuq_vlm.hazard.aleatoric import aleatoric_variance, heteroscedastic_loss
from policyuq_vlm.hazard.decomposition import decompose
from policyuq_vlm.hazard.epistemic import epistemic_variance
from policyuq_vlm.hazard.triage import BENIGN, REFER, SUSPICIOUS, triage_decision


def test_triage_truth_table() -> None:
    prob = torch.tensor([0.2, 0.8, 0.9, 0.1])
    total = torch.tensor([0.1, 0.1, 0.9, 0.9])
    decision = triage_decision(prob, total, delta=0.5)
    assert decision.tolist() == [BENIGN, SUSPICIOUS, REFER, REFER]


def test_decomposition_is_additive() -> None:
    ale = torch.tensor([0.2, 0.5])
    epi = torch.tensor([0.3, 0.1])
    budget = decompose(ale, epi)
    assert torch.allclose(budget.total, ale + epi)


def test_epistemic_variance_zero_when_constant() -> None:
    passes = torch.full((5, 3), 0.7)
    estimate = epistemic_variance(passes)
    assert torch.allclose(estimate.variance, torch.zeros(3), atol=1e-6)
    assert torch.allclose(estimate.mean, torch.full((3,), 0.7))


def test_heteroscedastic_loss_positive_and_finite() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 2)
    log_var = torch.zeros(8)
    target = torch.randint(0, 2, (8,))
    loss = heteroscedastic_loss(logits, log_var, target, samples=16)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_aleatoric_variance_matches_exp() -> None:
    log_var = torch.tensor([0.0, 1.0])
    assert torch.allclose(aleatoric_variance(log_var), torch.tensor([1.0, torch.e]))
