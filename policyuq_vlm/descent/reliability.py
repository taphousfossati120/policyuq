from __future__ import annotations

from policyuq_vlm.cairns.calibration import reliability_bins
from policyuq_vlm.descent.inference import Predictions


def reliability_diagram(predictions: Predictions, bins: int = 15) -> dict[str, list[float]]:
    confidence, accuracy = reliability_bins(predictions.prob, predictions.label, bins)
    return {"confidence": confidence, "accuracy": accuracy}
