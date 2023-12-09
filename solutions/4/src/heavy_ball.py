from typing import cast, Iterator

import torch as t
from torch.nn.parameter import Parameter


class HeavyBall:
    """\
    Based on [this blogpost](https://boostedml.com/2020/07/gradient-descent-and-momentum-the-heavy-ball-method.html).
    """

    def __init__(self, params: Iterator[Parameter], lr: float, momentum: float) -> None:
        self.params = list(params)
        self.prev_params = [t.zeros_like(p, requires_grad=False) for p in self.params]
        self.n_params = len(self.params)
        self.lr = lr
        self.momentum = momentum

    def step(self) -> None:
        curr_params = [p.detach().clone() for p in self.params]
        with t.no_grad():
            for i, prev_p in enumerate(self.prev_params):
                self.params[i] += -self.lr * self.params[i].grad + self.momentum * (
                    self.params[i] - prev_p
                )
        self.prev_params = curr_params

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
