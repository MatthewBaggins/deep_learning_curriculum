import torch as t
import torch.nn.functional as F


class ContrastiveCNN(t.nn.Module):
    """Simple 2-layer CNN for MNIST"""

    def __init__(self, d_embed: int = 64) -> None:
        super().__init__()
        self.d_embed = d_embed

        CONV_KERNEL_SIZE = 5
        CONV_STRIDE = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2

        self.conv1 = t.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDE,
            padding="same",
        )
        self.pool1 = t.nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        self.conv2 = t.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDE,
            padding="same",
        )
        self.pool2 = t.nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        self.fc = t.nn.Linear(in_features=7 * 7 * 64, out_features=d_embed)

    def forward(self, x: t.Tensor) -> t.Tensor:  # [batch 28 28]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def make_label_embeds(self, x: t.Tensor, y: t.Tensor) -> None:
        self.label_embeds = t.zeros(10, self.d_embed)
        with t.no_grad():
            x_embeds = self(x)
            for label in range(10):
                self.label_embeds[label] = x_embeds[y == label].mean(0).squeeze()

    # TODO: implement this method
    # def predict(self, x: t.Tensor) -> t.Tensor:
    #     with t.no_grad():
    #         embeds = self(x)
