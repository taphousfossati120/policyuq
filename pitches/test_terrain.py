from __future__ import annotations

import torch

from policyuq_vlm.terrain.labels import descriptor_target
from policyuq_vlm.terrain.vocabulary import VOCAB_SIZE, build_vocabulary


def test_vocabulary_is_eighty_seven() -> None:
    vocab = build_vocabulary()
    assert len(vocab) == VOCAB_SIZE == 87
    assert len({d.text for d in vocab}) == 87


def test_vocabulary_groups_present() -> None:
    groups = {d.group for d in build_vocabulary()}
    assert {"location", "signal_t2", "signal_dwi", "associated_epe", "associated_svi"} <= groups


def test_descriptor_target_in_range() -> None:
    pirads = torch.tensor([1, 3, 5])
    targets = descriptor_target(pirads)
    assert torch.all(targets >= 0)
    assert torch.all(targets < 87)
    assert targets[2] - targets[0] == 4
