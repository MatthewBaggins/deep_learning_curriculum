from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
import json
from pathlib import Path

import torch as t
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from . import CNN, Dataset


def train(
    model: CNN,
    ds: Dataset,
    optimizer: t.optim.Optimizer,
    *,
    n_epochs: int = 10,
    verbose: bool = True,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None
) -> TrainingHistory:
    save_path = Path("results")
    if not save_path.exists():
        save_path.mkdir()
    timestamp = datetime.now().isoformat("T", "minutes").replace(":", "-")
    
    th = TrainingHistory(timestamp)
    loss_fn = t.nn.CrossEntropyLoss()
    
    for epoch_i in range(1, n_epochs + 1):
        optimizer.zero_grad()
        
        train_logits = model(ds.train_x)
        train_loss = loss_fn(train_logits, ds.train_y)
        train_acc = acc_fn(train_logits, ds.train_y)
        with t.no_grad():
            test_logits = model(ds.test_x)
            test_loss = loss_fn(test_logits, ds.test_y)
            test_acc = acc_fn(test_logits, ds.test_y)
        train_loss.backward()
        optimizer.step()
        
        if isinstance(scheduler, LRScheduler):
            scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(train_loss)
        
        train_loss = train_loss.item()
        test_loss = test_loss.item()
        th.append(train_loss=train_loss, test_loss=test_loss, train_acc=train_acc, test_acc=test_acc)
        if verbose:
            print(f"Epoch {epoch_i} / {n_epochs}")
            print(f"\t{train_loss=:.4f}, {test_loss=:.4f}, {train_acc=:.2%}, {test_acc=:.2%}")
            
    t.save(model, save_path / th.model_filename)
    th.save_as_json(save_path / th.th_filename)
    
    return th


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    timestamp: str
    train_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    test_accuracies: list[float] = field(default_factory=list)

    def append(
        self, *, train_loss: float, test_loss: float, train_acc: float, test_acc: float
    ) -> None:
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)

    def save_as_json(self, path: Path) -> None:
        assert str(path).endswith(".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)
            
    @classmethod
    def load_from_json(cls, path: Path) -> TrainingHistory:
        assert path.exists()
        assert str(path).endswith(".json")
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)
        assert all(f.name in loaded for f in fields(cls))
        return cls(**loaded)

    @property
    def model_filename(self) -> str:
        return f"model_{self.timestamp}.pt"

    @property
    def th_filename(self) -> str:
        return f"th_{self.timestamp}.json"
    
    @property
    def n_epochs(self) -> int:
        return len(self)
    
    def __len__(self) -> int:
        return len(self.train_losses)

def acc_fn(
    logits: t.Tensor,
    target: t.Tensor,
) -> float:
    preds = logits.argmax(-1)
    acc = (preds == target).to(dtype=t.float).mean().item()
    return acc
