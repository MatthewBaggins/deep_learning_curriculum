from dataclasses import dataclass
from operator import eq
import pathlib

import torch as t
from torchvision.datasets import MNIST
from typing_extensions import Self

DATA_PATH = pathlib.Path("../../data")
assert DATA_PATH.exists()


@dataclass(frozen=True, slots=True)
class Dataset:
    train_x: t.Tensor
    train_y: t.Tensor
    test_x: t.Tensor
    test_y: t.Tensor
    dataset_fraction: float
    
    @classmethod
    def load(cls, dataset_fraction: float) -> Self:
        assert isinstance(dataset_fraction, float)
        assert 0 < dataset_fraction <= 1
        train = MNIST(str(DATA_PATH), train=True, download=True)
        test = MNIST(str(DATA_PATH), train=False, download=True)
        dataset_size = int(dataset_fraction * len(train.data))
        train_x = preprocess_batch(train.data)[:dataset_size]
        train_y = train.targets[:dataset_size]
        assert len(train_x) == len(train_y) == dataset_size
        test_x = preprocess_batch(test.data)
        test_y = test.targets
        return cls(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            dataset_fraction=dataset_fraction,
        )
        
    @property
    def dataset_size(self) -> int:
        return len(self.train_x)

    def __post_init__(self) -> None:
        assert len(self.train_x) == len(self.train_y)
        assert len(self.test_x) == len(self.test_y)
        assert self.train_x.ndim == self.test_x.ndim == 4
        assert self.train_y.ndim == self.test_y.ndim == 1
        assert self.train_x.shape[1:] == self.test_x.shape[1:]

def preprocess_batch(batch: t.Tensor) -> t.Tensor:
    assert batch.ndim == 3
    assert eq(*batch.shape[1:])
    batch_dim, im_dim = batch.shape[:2]
    processed_batch = batch.to(dtype=t.float32).unsqueeze(-1).reshape(batch_dim, 1, im_dim, im_dim)
    return (processed_batch - processed_batch.mean()) / processed_batch.std()
        