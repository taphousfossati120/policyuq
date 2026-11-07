from __future__ import annotations

import hashlib
from typing import cast

import torch
from torch import nn

from policyuq_vlm.terrain.vocabulary import Descriptor


def _hash_vector(text: str, dim: int) -> torch.Tensor:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(dim, generator=generator)


class FrozenDescriptorEncoder(nn.Module):
    def __init__(self, descriptors: list[Descriptor], dim: int) -> None:
        super().__init__()
        table = torch.stack([_hash_vector(d.text, dim) for d in descriptors])
        table = nn.functional.normalize(table, dim=-1)
        self.register_buffer("embeddings", table)
        self.dim = dim

    def forward(self) -> torch.Tensor:
        return cast(torch.Tensor, self.embeddings)


def load_pubmedbert_table(
    descriptors: list[Descriptor], dim: int, model_name: str
) -> torch.Tensor | None:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for descriptor in descriptors:
            tokens = tokenizer(descriptor.text, return_tensors="pt", truncation=True)
            pooled = model(**tokens).last_hidden_state.mean(dim=1)[0]
            rows.append(pooled)
    table = torch.stack(rows)
    if table.shape[-1] != dim:
        table = nn.functional.linear(table, torch.eye(dim, table.shape[-1]))
    return nn.functional.normalize(table, dim=-1)
