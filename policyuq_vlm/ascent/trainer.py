from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
from torch.utils.data import DataLoader

from policyuq_vlm.ascent.calibration import CalibrationTarget
from policyuq_vlm.ascent.optim import build_optimizer, clip_gradients
from policyuq_vlm.basecamp.config import ExperimentConfig
from policyuq_vlm.basecamp.logbook import get_logger
from policyuq_vlm.cairns.discrimination import auroc
from policyuq_vlm.hazard.aleatoric import heteroscedastic_loss
from policyuq_vlm.ledger.ppo import dpo_loss, ppo_surrogate, reinforce_loss
from policyuq_vlm.ledger.reward import terminal_reward
from policyuq_vlm.ledger.vision_language import vision_language_loss
from policyuq_vlm.model import PolicyUQVLM
from policyuq_vlm.route.advantage import generalized_advantage
from policyuq_vlm.route.state import EvidenceLibrary
from policyuq_vlm.terrain.datasets import MpMriDataset
from policyuq_vlm.terrain.labels import descriptor_target
from policyuq_vlm.terrain.records import Batch, MpMriCase, collate

_LOG = get_logger("ascent")


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        t2w=batch["t2w"].to(device),
        dwi=batch["dwi"].to(device),
        adc=batch["adc"].to(device),
        bbox=batch["bbox"].to(device),
        pirads=batch["pirads"].to(device),
        label=batch["label"].to(device),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class ThreePhaseTrainer:
    def __init__(self, model: PolicyUQVLM, cfg: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.calibration = CalibrationTarget()

    def loader(self, dataset: MpMriDataset, shuffle: bool) -> DataLoader[MpMriCase]:
        return DataLoader(
            dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    def batches(self, dataset: MpMriDataset, shuffle: bool) -> Iterator[Batch]:
        for raw in self.loader(dataset, shuffle):
            yield _to_device(cast(Batch, raw), self.device)

    def phase1_step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        if not self.cfg.ablation.use_vl_alignment:
            return 0.0
        aligned, table = self.model.descriptor_alignment(batch)
        targets = descriptor_target(batch["pirads"])
        loss = vision_language_loss(aligned, table, targets, self.cfg.encoder.temperature)
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(self.model.parameters(), self.cfg.optim.grad_clip)
        optimizer.step()
        return float(loss.item())

    def phase2_step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        logits, log_var = self.model.classify_fused(batch)
        loss = heteroscedastic_loss(
            logits, log_var, batch["label"], self.cfg.uncertainty.mc_samples
        )
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(self.model.parameters(), self.cfg.optim.grad_clip)
        optimizer.step()
        return float(loss.item())

    def init_calibration(self, batch: Batch) -> None:
        with torch.no_grad():
            logits, _ = self.model.classify_fused(batch)
            prob = torch.softmax(logits, dim=-1)
            confidence, predicted = prob.max(dim=-1)
            correct = predicted == batch["label"]
        self.calibration.update(confidence.cpu(), correct.cpu())

    def phase3_round(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        with torch.no_grad():
            out = self.model.forward(batch, stochastic=True)
            prob = torch.softmax(out.logits, dim=-1)
            confidence, predicted = prob.max(dim=-1)
            p_ideal = self.calibration.lookup(confidence.cpu()).to(self.device)
            reward = terminal_reward(
                predicted,
                batch["label"],
                confidence,
                p_ideal,
                out.trajectory.lengths,
                self.cfg.reward,
            )
            traj = out.trajectory
            library = EvidenceLibrary(out.library.evidence.detach())
            steps = traj.states.shape[1]
            reward_steps = torch.zeros(traj.states.shape[0], steps, device=self.device)
            terminal = (traj.lengths - 1).clamp(0, steps - 1)
            reward_steps[torch.arange(reward_steps.shape[0]), terminal] = reward
            advantages, returns = generalized_advantage(
                reward_steps, traj.values, traj.mask, self.cfg.ppo.gamma, self.cfg.ppo.gae_lambda
            )
            states = traj.states.detach()
            actions = traj.actions
            old_logp = traj.logprobs
            mask = traj.mask

        algorithm = self.cfg.ablation.policy_algorithm
        total_loss = 0.0
        for _ in range(self.cfg.ppo.epochs_per_round):
            new_logp, entropy, values = self.model.reasoner.evaluate_actions(
                library, states, actions
            )
            if algorithm == "ppo":
                losses = ppo_surrogate(
                    new_logp,
                    old_logp,
                    advantages,
                    returns,
                    values,
                    entropy,
                    mask,
                    self.cfg.ppo.clip,
                )
                policy = losses.policy
                value = losses.value
                entropy_term = losses.entropy
            else:
                if algorithm == "reinforce":
                    policy = reinforce_loss(new_logp, advantages, mask)
                elif algorithm == "dpo":
                    policy = dpo_loss(new_logp, old_logp, advantages, mask)
                else:
                    raise ValueError(f"unknown policy algorithm '{algorithm}'")
                value = _masked_mean((returns - values).pow(2), mask)
                entropy_term = _masked_mean(entropy, mask)

            het_logits, het_log_var = self.model.classify_fused(batch)
            het = heteroscedastic_loss(
                het_logits, het_log_var, batch["label"], self.cfg.uncertainty.mc_samples
            )
            total = (
                policy
                + value
                - self.cfg.ppo.entropy * entropy_term
                + self.cfg.phases.het_weight * het
            )
            optimizer.zero_grad()
            total.backward()
            clip_gradients(self.model.parameters(), self.cfg.optim.grad_clip)
            optimizer.step()
            total_loss += float(total.item())
        return total_loss / self.cfg.ppo.epochs_per_round

    def validation_auc(self, dataset: MpMriDataset) -> float:
        self.model.eval()
        probs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        with torch.no_grad():
            for moved in self.batches(dataset, shuffle=False):
                out = self.model.forward(moved, stochastic=False)
                probs.append(out.prob.cpu())
                labels.append(moved["label"].cpu())
        self.model.train()
        return auroc(torch.cat(probs), torch.cat(labels))

    def fit(
        self,
        train: MpMriDataset,
        val: MpMriDataset,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        encoder_opt = build_optimizer(self.model.encoder.parameters(), self.cfg.optim)
        head_params = list(self.model.encoder.parameters()) + list(self.model.head.parameters())
        phase2_opt = build_optimizer(iter(head_params), self.cfg.optim)
        phase3_opt = build_optimizer(
            self.model.parameters(), self.cfg.optim, lr_scale=self.cfg.phases.policy_lr_scale
        )

        self.model.train()
        for _ in range(self.cfg.phases.pretrain_epochs):
            for i, batch in enumerate(self.batches(train, shuffle=True)):
                if max_batches is not None and i >= max_batches:
                    break
                self.phase1_step(batch, encoder_opt)

        for _ in range(self.cfg.phases.finetune_epochs):
            for i, batch in enumerate(self.batches(train, shuffle=True)):
                if max_batches is not None and i >= max_batches:
                    break
                self.phase2_step(batch, phase2_opt)

        for batch in self.batches(train, shuffle=False):
            self.init_calibration(batch)
            break

        best_auc = 0.0
        patience = 0
        for epoch in range(self.cfg.phases.policy_epochs):
            for i, batch in enumerate(self.batches(train, shuffle=True)):
                if max_batches is not None and i >= max_batches:
                    break
                self.phase3_round(batch, phase3_opt)
            if (epoch + 1) % self.cfg.phases.calibration_refresh == 0:
                for batch in self.batches(train, shuffle=False):
                    self.init_calibration(batch)
                    break
            current = self.validation_auc(val)
            if current > best_auc + 1e-4:
                best_auc = current
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.phases.early_stop_patience:
                break
        return {"val_auc": best_auc}
