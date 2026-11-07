from __future__ import annotations

from policyuq_vlm.cairns.discrimination import auroc
from policyuq_vlm.cairns.statistics import bootstrap_ci
from policyuq_vlm.descent.inference import Predictions


def per_pirads_auc(predictions: Predictions) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for grade in range(1, 6):
        selection = predictions.pirads == grade
        count = int(selection.sum().item())
        if count == 0:
            continue
        prob = predictions.prob[selection]
        label = predictions.label[selection]
        low, high = bootstrap_ci(prob, label)
        result[grade] = {
            "n": float(count),
            "auroc": auroc(prob, label),
            "ci_low": low,
            "ci_high": high,
        }
    return result


def equivocal_group(predictions: Predictions) -> dict[str, float]:
    selection = predictions.pirads == 3
    if int(selection.sum().item()) == 0:
        return {"n": 0.0, "auroc": 0.5}
    prob = predictions.prob[selection]
    label = predictions.label[selection]
    return {"n": float(int(selection.sum().item())), "auroc": auroc(prob, label)}
