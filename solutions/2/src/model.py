import torch as t
from torch import nn
from torch.nn import functional as F

KERNEL_SIZE = 3
FC_DIM = 64

class CNN(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=n_channels, kernel_size=KERNEL_SIZE, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=KERNEL_SIZE, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(n_channels * 7 * 7, FC_DIM)
        self.fc2 = nn.Linear(FC_DIM, 10)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
