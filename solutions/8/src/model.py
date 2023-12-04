import torch as t
from torch import nn
from torch.nn import functional as F

N_CHANNELS = 6
FC_DIM = 64
CONV_KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 2


class CNN(nn.Module):
    def __init__(self, n_channels: int = N_CHANNELS, fc_dim: int = FC_DIM) -> None:
        super().__init__()
        
        self.n_channels = n_channels
        self.fc_dim = fc_dim
        
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=n_channels, kernel_size=CONV_KERNEL_SIZE, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE)
        self.conv2 = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=CONV_KERNEL_SIZE, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE)
        self.fc1 = nn.Linear(n_channels * 7 * 7, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 10)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x_post_conv1 = self.pool1(F.relu(self.conv1(x)))
        x_post_conv2 = self.pool2(F.relu(self.conv2(x_post_conv1)))
        x_post_fc1 = F.relu(self.fc1(x_post_conv2.flatten(1)))
        x_post_fc2 = self.fc2(x_post_fc1)
        return x_post_fc2
