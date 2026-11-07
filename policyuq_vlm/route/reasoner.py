from __future__ import annotations

import torch
from attrs import frozen
from torch import nn
from torch.distributions import Categorical

from policyuq_vlm.basecamp.config import PolicyConfig
from policyuq_vlm.route.agent import PolicyAgent
from policyuq_vlm.route.context import ContextGRU
from policyuq_vlm.route.state import EvidenceLibrary

UNCERTAINTY_DIM = 2


@frozen
class Trajectory:
    states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    entropies: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    final_state: torch.Tensor


class Reasoner(nn.Module):
    def __init__(self, cfg: PolicyConfig, fused_dim: int, token_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_dim = fused_dim + cfg.gru_hidden + UNCERTAINTY_DIM
        self.gru = ContextGRU(token_dim, cfg.gru_hidden, cfg.gru_layers)
        self.agent = PolicyAgent(self.state_dim, token_dim, cfg.policy_hidden)
        self.uncertainty_proxy = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.gru_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.gru_hidden // 2, UNCERTAINTY_DIM),
            nn.Softplus(),
        )

    def rollout(
        self,
        library: EvidenceLibrary,
        fused: torch.Tensor,
        stochastic: bool,
    ) -> Trajectory:
        batch = library.batch_size
        device = fused.device
        hiddens = self.gru.initial(batch, device)
        uncertainty = torch.zeros(batch, UNCERTAINTY_DIM, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        final_state = torch.zeros(batch, self.state_dim, device=device)

        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        active_steps: list[torch.Tensor] = []

        for t in range(self.cfg.max_steps):
            context = self.gru.readout(hiddens)
            state_vec = torch.cat([fused, context, uncertainty], dim=-1)
            logits, value = self.agent(state_vec, library)
            dist = Categorical(logits=logits)
            action = dist.sample() if stochastic else logits.argmax(dim=-1)
            active = ~done

            commit_now = (action == library.commit_index) & active
            last_step = active & torch.full_like(done, t == self.cfg.max_steps - 1)
            finalize = commit_now | last_step
            final_state = torch.where(finalize.unsqueeze(-1), state_vec, final_state)

            states.append(state_vec)
            actions.append(action)
            logprobs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())
            active_steps.append(active.float())
            lengths = lengths + active.long()

            token = library.feature(action)
            advanced = self.gru.step(hiddens, token)
            update = (active & ~commit_now).unsqueeze(-1)
            hiddens = [
                torch.where(update, nh, oh) for nh, oh in zip(advanced, hiddens, strict=True)
            ]
            proxy = self.uncertainty_proxy(self.gru.readout(hiddens))
            uncertainty = torch.where(active.unsqueeze(-1), proxy, uncertainty)
            done = done | commit_now | last_step
            if bool(done.all()):
                break

        return Trajectory(
            states=torch.stack(states, dim=1),
            actions=torch.stack(actions, dim=1),
            logprobs=torch.stack(logprobs, dim=1),
            values=torch.stack(values, dim=1),
            entropies=torch.stack(entropies, dim=1),
            mask=torch.stack(active_steps, dim=1),
            lengths=lengths,
            final_state=final_state,
        )

    def evaluate_actions(
        self, library: EvidenceLibrary, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = states.shape[1]
        logp: list[torch.Tensor] = []
        ent: list[torch.Tensor] = []
        val: list[torch.Tensor] = []
        for t in range(steps):
            logits, value = self.agent(states[:, t], library)
            dist = Categorical(logits=logits)
            logp.append(dist.log_prob(actions[:, t]))
            ent.append(dist.entropy())
            val.append(value)
        return (
            torch.stack(logp, dim=1),
            torch.stack(ent, dim=1),
            torch.stack(val, dim=1),
        )
