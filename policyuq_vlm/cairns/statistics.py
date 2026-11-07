from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from policyuq_vlm.cairns.discrimination import auroc


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    result = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        result[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = result
    return out


def delong_test(
    prob_a: torch.Tensor, prob_b: torch.Tensor, label: torch.Tensor
) -> tuple[float, float]:
    y = label.detach().cpu().numpy().astype(int)
    positives = y == 1
    negatives = ~positives
    scores = np.stack([prob_a.detach().cpu().numpy(), prob_b.detach().cpu().numpy()])
    m = int(positives.sum())
    n = int(negatives.sum())
    if m == 0 or n == 0:
        return 0.0, 1.0
    pos = scores[:, positives]
    neg = scores[:, negatives]
    tx = np.stack([_midrank(pos[k]) for k in range(2)])
    ty = np.stack([_midrank(neg[k]) for k in range(2)])
    tz = np.stack([_midrank(scores[k]) for k in range(2)])
    aucs = tz[:, positives].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, positives] - tx) / n
    v10 = 1.0 - (tz[:, negatives] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(diff), 1.0
    z = diff / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def bootstrap_ci(
    prob: torch.Tensor, label: torch.Tensor, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = prob.shape[0]
    estimates: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_prob = prob[idx]
        sample_label = label[idx]
        if len(torch.unique(sample_label)) < 2:
            continue
        estimates.append(auroc(sample_prob, sample_label))
    if not estimates:
        point = auroc(prob, label)
        return point, point
    low = float(np.quantile(estimates, alpha / 2))
    high = float(np.quantile(estimates, 1 - alpha / 2))
    return low, high
