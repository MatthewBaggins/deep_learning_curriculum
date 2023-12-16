from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from operator import eq
import pathlib

import einops
import torch as t
from torchvision.datasets import MNIST
from tqdm import tqdm

DATA_PATH = pathlib.Path("../../data")
assert DATA_PATH.exists()


BatchX = BatchY = DiffPairInds = SamePairInds = t.Tensor
TrainBatch = tuple[BatchX, BatchY, DiffPairInds, SamePairInds]


@dataclass(frozen=True, slots=True)
class Dataset:
    train_x: t.Tensor
    train_y: t.Tensor
    test_x: t.Tensor
    test_y: t.Tensor

    @classmethod
    def load(cls) -> Dataset:
        train = MNIST(str(DATA_PATH), train=True, download=True)
        test = MNIST(str(DATA_PATH), train=False, download=True)
        train_x = _preprocess_batch(train.data)
        train_y = train.targets
        test_x = _preprocess_batch(test.data)
        test_y = test.targets
        return cls(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
        )

    @cache
    def get_train_batches(self, batch_size: int = 32) -> list[TrainBatch]:
        """`TrainBatch = tuple[BatchX, BatchY, DiffPairInds, SamePairInds]`"""
        n_batches = len(self.train_x) // batch_size
        batches = [
            (
                self.train_x[batch_i * batch_size : (batch_i + 1) * batch_size],
                batch_y := self.train_y[
                    batch_i * batch_size : (batch_i + 1) * batch_size
                ],
                *get_diff_and_sim_pair_inds(batch_y),
            )
            for batch_i in tqdm(
                range(n_batches), desc=f"Generating train batches for {batch_size=}"
            )
        ]
        return batches

    # def get_train_batch_gen(
    #     self, batch_size: int = 32
    # ) -> Iterable[tuple[t.Tensor, t.Tensor]]:
    #     n_batches = len(self.train_x) // batch_size
    #     return (
    #         (
    #             self.train_x[batch_i * batch_size : (batch_i + 1) * batch_size],
    #             self.train_y[batch_i * batch_size : (batch_i + 1) * batch_size],
    #         )
    #         for batch_i in range(n_batches)
    #     )


def get_diff_and_sim_pair_inds(y: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    assert y.ndim == 1
    assert y.dtype == t.int64
    y_rep = einops.repeat(y, "b -> b rep", rep=len(y))
    diff_pair_inds = (y_rep != y_rep.T).nonzero()
    same_pair_inds = (y_rep == y_rep.T).nonzero()
    return diff_pair_inds, same_pair_inds


def _preprocess_batch(batch: t.Tensor) -> t.Tensor:
    assert batch.ndim == 3
    assert eq(*batch.shape[1:])
    batch_dim, im_dim = batch.shape[:2]
    processed_batch = (
        batch.to(dtype=t.float32).unsqueeze(-1).reshape(batch_dim, 1, im_dim, im_dim)
    )
    return (processed_batch - processed_batch.mean()) / processed_batch.std()
