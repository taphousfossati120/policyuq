from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def auroc(prob: torch.Tensor, label: torch.Tensor) -> float:
    y = _to_numpy(label).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, _to_numpy(prob)))


def average_precision(prob: torch.Tensor, label: torch.Tensor) -> float:
    y = _to_numpy(label).astype(int)
    if len(np.unique(y)) < 2:
        return float(y.mean())
    return float(average_precision_score(y, _to_numpy(prob)))


def sensitivity_at_specificity(
    prob: torch.Tensor, label: torch.Tensor, specificity: float = 0.90
) -> float:
    y = _to_numpy(label).astype(int)
    if len(np.unique(y)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y, _to_numpy(prob))
    feasible = np.where(fpr <= (1.0 - specificity))[0]
    if feasible.size == 0:
        return 0.0
    return float(tpr[feasible].max())
