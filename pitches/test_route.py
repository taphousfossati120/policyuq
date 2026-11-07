from __future__ import annotations

import torch

from policyuq_vlm.basecamp.config import PolicyConfig
from policyuq_vlm.route.context import ContextGRU
from policyuq_vlm.route.reasoner import Reasoner
from policyuq_vlm.route.state import EvidenceLibrary


def test_evidence_action_space() -> None:
    library = EvidenceLibrary(torch.randn(2, 9, 8))
    assert library.n_tokens == 9
    assert library.commit_index == 9
    assert library.n_actions == 10


def test_evidence_feature_gather() -> None:
    evidence = torch.randn(2, 5, 4)
    library = EvidenceLibrary(evidence)
    feature = library.feature(torch.tensor([2, 0]))
    assert torch.allclose(feature[0], evidence[0, 2])
    assert torch.allclose(feature[1], evidence[1, 0])


def test_context_gru_step_shape() -> None:
    gru = ContextGRU(input_dim=4, hidden=6, layers=2)
    hiddens = gru.initial(3, torch.device("cpu"))
    updated = gru.step(hiddens, torch.randn(3, 4))
    assert len(updated) == 2
    assert gru.readout(updated).shape == (3, 6)


def test_rollout_respects_max_steps() -> None:
    cfg = PolicyConfig(gru_hidden=8, policy_hidden=8, max_steps=4)
    reasoner = Reasoner(cfg, fused_dim=8, token_dim=8)
    library = EvidenceLibrary(torch.randn(2, 6, 8))
    traj = reasoner.rollout(library, torch.randn(2, 8), stochastic=True)
    assert traj.states.shape[1] <= cfg.max_steps
    assert torch.all(traj.lengths >= 1)
    assert traj.final_state.shape == (2, reasoner.state_dim)


def test_evaluate_actions_matches_steps() -> None:
    cfg = PolicyConfig(gru_hidden=8, policy_hidden=8, max_steps=3)
    reasoner = Reasoner(cfg, fused_dim=8, token_dim=8)
    library = EvidenceLibrary(torch.randn(2, 6, 8))
    traj = reasoner.rollout(library, torch.randn(2, 8), stochastic=True)
    logp, ent, val = reasoner.evaluate_actions(library, traj.states, traj.actions)
    assert logp.shape == traj.actions.shape
    assert ent.shape == traj.actions.shape
    assert val.shape == traj.actions.shape
