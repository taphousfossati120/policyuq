from __future__ import annotations

from policyuq_vlm.cairns.calibration import brier_score, expected_calibration_error
from policyuq_vlm.cairns.discrimination import (
    auroc,
    average_precision,
    sensitivity_at_specificity,
)
from policyuq_vlm.descent.inference import Predictions


def evaluate_benchmark(predictions: Predictions) -> dict[str, float]:
    return {
        "auroc": auroc(predictions.prob, predictions.label),
        "ap": average_precision(predictions.prob, predictions.label),
        "sens_at_90": sensitivity_at_specificity(predictions.prob, predictions.label, 0.90),
        "ece": expected_calibration_error(predictions.prob, predictions.label, 15),
        "brier": brier_score(predictions.prob, predictions.label),
    }
