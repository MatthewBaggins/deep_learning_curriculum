from __future__ import annotations

from dataclasses import dataclass
from operator import eq
from pathlib import Path
from typing import Literal

import torch as t
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST

DATA_PATH = Path("")
while not (DATA_PATH / "data").exists():
    DATA_PATH = ".." / DATA_PATH
else:
    DATA_PATH = DATA_PATH / "data"


class SimpleCNN(t.nn.Module):
    """Simple 2-layer CNN for MNIST"""

    def __init__(self) -> None:
        super().__init__()
        CONV_KERNEL_SIZE = 5
        CONV_STRIDE = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2
        PADDING1 = (CONV_KERNEL_SIZE - 1) // 2
        PADDING2 = (CONV_KERNEL_SIZE - 1) // 2

        self.conv1 = t.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDE,
            padding=PADDING1,  # "same",
        )
        self.pool1 = t.nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        self.conv2 = t.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDE,
            padding=PADDING2,  # "same",
        )
        self.pool2 = t.nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        self.fc = t.nn.Linear(in_features=7 * 7 * 64, out_features=1024)

    def forward(self, x: t.Tensor) -> t.Tensor:  # [batch 28 28]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.fc(x)
        return x


@dataclass(frozen=True, slots=True)
class DatasetSimpleCNN:
    train_x: t.Tensor
    train_y: t.Tensor
    test_x: t.Tensor
    test_y: t.Tensor
    fashion_mnist: bool

    @classmethod
    def load(cls, *, fashion_mnist: bool = False) -> DatasetSimpleCNN:
        dataset = FashionMNIST if fashion_mnist else MNIST
        train = dataset(str(DATA_PATH), train=True, download=True)
        test = dataset(str(DATA_PATH), train=False, download=True)
        train_x = _preprocess_batch(train.data)
        train_y = train.targets
        test_x = _preprocess_batch(test.data)
        test_y = test.targets
        return cls(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            fashion_mnist=fashion_mnist,
        )


def _preprocess_batch(batch: t.Tensor) -> t.Tensor:
    assert batch.ndim == 3
    assert eq(*batch.shape[1:])
    batch_dim, im_dim = batch.shape[:2]
    processed_batch = (
        batch.to(dtype=t.float32).unsqueeze(-1).reshape(batch_dim, 1, im_dim, im_dim)
    )
    return (processed_batch - processed_batch.mean()) / processed_batch.std()


def main() -> None:
    model = SimpleCNN()
    ds_m = DatasetSimpleCNN.load()
    model(ds_m.train_x[:100])


if __name__ == "__main__":
    main()
