from dataclasses import dataclass, field

import torch as t
from torch import nn, optim

def loss_fn(logits: t.Tensor, tokens: t.Tensor) -> t.Tensor:
    logits = logits[:, :-1]
    tokens = tokens[:, 1:].unsqueeze(-1)
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens)[..., 0]
    return -correct_log_probs.mean()

def acc_fn(logits: t.Tensor, tokens: t.Tensor) -> float:
    n_ctx = logits.size(1) // 2
    logits = logits[:, n_ctx:-1]
    preds = logits.argmax(-1)
    tokens = tokens[:, n_ctx+1:]
    acc = (preds == tokens).mean(dtype=t.float).item()
    return acc

@dataclass(frozen=True, slots=True)
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    test_accuracies: list[float] = field(default_factory=list)

    def append(self, train_loss: float, test_loss: float, train_acc: float, test_acc: float) -> None:
        assert isinstance(train_loss, float)
        assert isinstance(test_loss, float)
        assert isinstance(train_acc, float)
        assert isinstance(test_acc, float)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)

def train(model: nn.Module, train_data, test_data, *, loss_fn, optimizer, n_epochs: int = 20, log_each_epochs: int | None = 10, scheduler = None) -> TrainingHistory:
    th = TrainingHistory()

    for epoch_i in range(n_epochs):
        epoch_train_losses = []
        epoch_train_accuracies = []
        
        for train_batch in train_data:
            # Forward
            train_logits = model(train_batch)
            # Loss
            train_loss = loss_fn(train_logits, train_batch)
            # Backward and update
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Accuracy
            train_acc = acc_fn(train_logits, train_batch)
            # Append
            epoch_train_losses.append(train_loss.item())
            epoch_train_accuracies.append(train_acc)
        
        with t.no_grad():
            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_data)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            elif scheduler is not None:
                scheduler.step()
            test_acc = acc_fn(test_logits, test_data)
        
        
        # Measure    
        train_loss = t.tensor(epoch_train_losses).mean().item()
        train_acc = t.tensor(epoch_train_accuracies).mean().item()
        th.append(train_loss=train_loss, test_loss=test_loss.item(), train_acc=train_acc, test_acc=test_acc)
        
        if log_each_epochs and epoch_i % log_each_epochs == 0:
            print(f"[{epoch_i}] \n    loss: train={train_loss:.3f}; test={test_loss:.3f};\n    acc:  train={train_acc:.2%}; test={test_acc:.2%}")
        
    return th