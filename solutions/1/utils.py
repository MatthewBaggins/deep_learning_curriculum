import re
from typing import cast

import torch as t
from torch.nn import functional as F

def pad_last_dim(x: t.Tensor, max_dim: int) -> t.Tensor:
    """Pad last dimension of the tensor"""
    assert x.ndim > 0
    assert x.size(-1) <= max_dim
    if x.size(-1) == max_dim:
        return x
    padding = t.zeros(*x.shape[:-1], max_dim - x.size(-1))
    return t.cat((x, padding), dim=-1)

def tokenize(text: str) -> list[str]:
    """Split text into words/tokens based on word boundaries (`\\b`)."""
    return re.split(r"\b", text)

def to_one_hot(x: t.Tensor, max_last_dim: int | None = None) -> t.Tensor:
    assert x.ndim == 2
    x_one_hot = F.one_hot(x)
    return pad_last_dim(x_one_hot, max_last_dim or x_one_hot.size(-1)).to(dtype=t.float32)

def is_one_hot(x: t.Tensor) -> bool:
    assert x.ndim >= 2
    return cast(bool, ((x != 0).sum(-1) == 1).all().item())
