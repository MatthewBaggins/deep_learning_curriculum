from dataclasses import dataclass, field
from typing import Self

import torch as t

from utils import to_one_hot, is_one_hot #pylint:disable=import-error


@dataclass(frozen=True, slots=True)
class Config:
    d_model: int
    d_vocab: int
    n_layers: int
    n_heads: int
    n_ctx: int
    dropout: float = field(default=0.0, kw_only=True, repr=False)
    epsilon: float = field(default=1e-6, kw_only=True, repr=False)
    ln_eps: float = 1e-6

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0
        assert 0 <= self.dropout <= 0.9, f"unreasonable dropout: {self.dropout}"

    @property
    def d_mlp(self) -> int:
        return self.d_model * 4

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def dummy(cls, **kwargs) -> Self:
        return cls(
            **(
                dict(d_model=32, n_ctx=256, d_vocab=512, n_layers=2, n_heads=4)
                | kwargs
            )
        )

    def random_resid(self, n_batches: int = 16) -> t.Tensor:
        return t.rand(n_batches, self.n_ctx, self.d_model)

    def random_tokens(
        self,
        n_batches: int = 16,
        n_ctx: int | None = None,
        max_token: int | None = None,
    ) -> t.Tensor:
        n_ctx = n_ctx or self.n_ctx
        max_token = max_token or self.d_vocab - 1
        tokens = (max_token * t.rand(n_batches, n_ctx)).round().to(dtype=t.int64)
        return tokens

    def random_tokens_one_hot(
        self, n_batches: int = 16, max_token: int | None = None
    ) -> t.Tensor:
        tokens = self.random_tokens(n_batches=n_batches, max_token=max_token)
        tokens_one_hot = to_one_hot(tokens, self.d_vocab)
        assert tokens_one_hot.shape[-2:] == (self.n_ctx, self.d_vocab)
        assert is_one_hot(tokens_one_hot)
        return tokens_one_hot
