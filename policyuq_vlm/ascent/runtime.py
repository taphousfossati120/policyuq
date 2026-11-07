from __future__ import annotations

import os
import tempfile
from typing import Any

import torch
from torch import nn


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def atomic_save(state: dict[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(handle)
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict[str, Any]:
    state: dict[str, Any] = torch.load(path, map_location=map_location)
    return state


class Ema:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
