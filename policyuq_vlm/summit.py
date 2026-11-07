from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from policyuq_vlm.ascent.runtime import atomic_save, load_checkpoint, select_device
from policyuq_vlm.ascent.trainer import ThreePhaseTrainer
from policyuq_vlm.basecamp.config import load_experiment
from policyuq_vlm.basecamp.logbook import get_logger
from policyuq_vlm.descent.benchmark import evaluate_benchmark
from policyuq_vlm.descent.inference import collect_predictions
from policyuq_vlm.descent.referral import referral_analysis
from policyuq_vlm.descent.subgroup import per_pirads_auc
from policyuq_vlm.expedition import build_model, build_splits, volume_shape
from policyuq_vlm.hazard.triage import select_threshold, triage_decision
from policyuq_vlm.model import PolicyUQVLM
from policyuq_vlm.terrain.records import Batch

_LOG = get_logger("summit")


class _TraceWrapper(nn.Module):
    def __init__(self, model: PolicyUQVLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, t2w: torch.Tensor, dwi: torch.Tensor, adc: torch.Tensor) -> torch.Tensor:
        batch = Batch(
            t2w=t2w,
            dwi=dwi,
            adc=adc,
            bbox=torch.zeros(t2w.shape[0], 6),
            pirads=torch.ones(t2w.shape[0], dtype=torch.long),
            label=torch.zeros(t2w.shape[0], dtype=torch.long),
        )
        logits, _ = self.model.classify_fused(batch)
        return torch.softmax(logits, dim=-1)[:, 1]


class Summit:
    def train(
        self,
        config: str,
        base: str | None = None,
        out: str = "runs/policyuq.pt",
        max_batches: int | None = None,
    ) -> dict[str, float]:
        cfg = load_experiment(config, base)
        train_set, val_set = build_splits(cfg)
        device = select_device()
        model = build_model(cfg)
        trainer = ThreePhaseTrainer(model, cfg, device)
        result = trainer.fit(train_set, val_set, max_batches=max_batches)
        atomic_save({"model": model.state_dict(), "seed": cfg.seed, "experiment": cfg.name}, out)
        _LOG.info("training complete: %s", result)
        return result

    def evaluate(
        self,
        config: str,
        base: str | None = None,
        checkpoint: str | None = None,
        passes: int | None = None,
    ) -> dict[str, Any]:
        cfg = load_experiment(config, base)
        _, val_set = build_splits(cfg)
        device = select_device()
        model = build_model(cfg)
        if checkpoint is not None:
            model.load_state_dict(load_checkpoint(checkpoint)["model"])
        model.to(device)
        passes = passes or cfg.uncertainty.stochastic_passes
        predictions = collect_predictions(model, val_set, device, passes)
        report = {
            "benchmark": evaluate_benchmark(predictions),
            "referral": referral_analysis(predictions, cfg.uncertainty.sensitivity_target),
            "per_pirads": per_pirads_auc(predictions),
        }
        _LOG.info("evaluation: %s", report["benchmark"])
        return report

    def infer(
        self,
        config: str,
        base: str | None = None,
        checkpoint: str | None = None,
        passes: int | None = None,
    ) -> dict[str, Any]:
        cfg = load_experiment(config, base)
        _, val_set = build_splits(cfg)
        device = select_device()
        model = build_model(cfg)
        if checkpoint is not None:
            model.load_state_dict(load_checkpoint(checkpoint)["model"])
        model.to(device)
        passes = passes or cfg.uncertainty.stochastic_passes
        predictions = collect_predictions(model, val_set, device, passes)
        delta = select_threshold(
            predictions.prob,
            predictions.total,
            predictions.label,
            cfg.uncertainty.sensitivity_target,
        )
        decisions = triage_decision(predictions.prob, predictions.total, delta)
        return {
            "probability": predictions.prob[:8].tolist(),
            "triage": decisions[:8].tolist(),
            "threshold": delta,
        }

    def trace(self, config: str, base: str | None = None, out: str = "runs/policyuq.onnx") -> str:
        cfg = load_experiment(config, base)
        model = build_model(cfg)
        wrapper = _TraceWrapper(model)
        wrapper.eval()
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        depth, height, width = volume_shape(cfg)
        example_input = torch.zeros(1, 1, depth, height, width)
        torch.onnx.export(
            wrapper,
            (example_input, example_input.clone(), example_input.clone()),
            out,
            input_names=["t2w", "dwi", "adc"],
            output_names=["probability"],
            opset_version=17,
            dynamo=False,
        )
        _LOG.info("exported ONNX graph to %s", out)
        return out


def main() -> None:
    import fire

    fire.Fire(Summit)


if __name__ == "__main__":
    main()
