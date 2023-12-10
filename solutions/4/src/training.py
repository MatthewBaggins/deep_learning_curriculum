from __future__ import annotations

from datetime import datetime
import itertools as it
import json
import math
import os
from pathlib import Path
from typing import Literal, TypedDict


import torch as t
from torch.optim import lr_scheduler, SGD, Adam

from .simple_cnn import SimpleCNN, DatasetSimpleCNN
from .utils import split_into_batches


def acc_fn(
    logits: t.Tensor,
    target: t.Tensor,
) -> float:
    preds = logits.argmax(-1)
    acc = (preds == target).to(dtype=t.float).mean().item()
    return acc


class TrainingResult(TypedDict):
    timestamp: str
    # names
    model_name: str
    dataset_name: str
    optimizer_name: str
    optimizer_args: dict[str, float | tuple[float, float]]
    scheduler_name: str
    # numbers
    batch_size: int
    target_accuracy: float
    acc_measure_freq: int
    train_losses: list[float]
    test_accuracies: list[float]
    steps_to_target: int


def get_optimizer_args(
    optimizer: SGD | Adam,
) -> dict[str, float | tuple[float, float]]:
    if isinstance(optimizer, SGD):
        return {
            "lr": optimizer.param_groups[0]["lr"],
            "momentum": optimizer.param_groups[0]["momentum"],
        }
    return {
        "lr": optimizer.param_groups[0]["lr"],
        "betas": optimizer.param_groups[0]["betas"],
    }


# Target accuract on the val/test set after which training terminates
TARGET_ACCURACY_M = 0.97
TARGET_ACCURACY_F = 0.90
# Maximum number of steps (batches) after which training terminates
MAX_N_STEPS = 2**14
# How often to measure accuracy, each 100 steps (batches) by default
ACC_MEASURE_FREQ = 20

RESULTS_PATH = Path()
while not "src" in os.listdir(RESULTS_PATH):
    RESULTS_PATH = ".." / RESULTS_PATH

RESULTS_PATH /= "results"
if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir()


def train(
    model: SimpleCNN,
    ds: DatasetSimpleCNN,
    optimizer: SGD | Adam,
    optimizer_name: str,
    batch_size: int,
    *,
    # target_accuracy: float = TARGET_ACCURACY_M,
    acc_measure_freq: int = ACC_MEASURE_FREQ,
    max_n_steps: int = MAX_N_STEPS,
    verbose: bool = True,
    scheduler: lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau | None = None,
    save_model: bool = False,
    save_tr: bool = True,
) -> tuple[SimpleCNN, TrainingResult]:
    loss_fn = t.nn.CrossEntropyLoss()
    target_accuracy = TARGET_ACCURACY_F if ds.fashion_mnist else TARGET_ACCURACY_M

    train_x_batches = split_into_batches(ds.train_x, batch_size)
    train_y_batches = split_into_batches(ds.train_y, batch_size)
    batch_iter = it.cycle(zip(train_x_batches, train_y_batches))

    train_losses: list[float] = []
    test_accuracies: list[float] = []

    for step_i, (batch_x, batch_y) in enumerate(batch_iter):
        if step_i >= max_n_steps:
            break
        optimizer.zero_grad()
        train_logits = model(batch_x)
        train_loss = loss_fn(train_logits, batch_y)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        if step_i % acc_measure_freq == 0:
            with t.no_grad():
                test_logits = model(ds.test_x)
                test_acc = acc_fn(test_logits, ds.test_y)
                test_accuracies.append(test_acc)
            running_loss = t.tensor(train_losses[-100:]).mean().item()
            if verbose:
                print(f"Step {step_i}: {running_loss=:.4f}, {test_acc=:.2%}")
            if test_acc >= target_accuracy:
                print(f"{target_accuracy=} achieved after {step_i} steps")
                break
            if step_i % 100 == 0:
                if isinstance(scheduler, lr_scheduler.LRScheduler):
                    scheduler.step()
                elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(running_loss)
            last_20_accs = t.tensor(test_accuracies[-20:])
            if len(last_20_accs) == 20 and len(last_20_accs.unique()) == 1:
                print(
                    f"Loss stagnant at {test_acc} for the last 20 checks. Ending training prematurely."
                )
                break

    timestamp = datetime.now().isoformat("T", "minutes").replace(":", "-")
    optimizer_args = get_optimizer_args(optimizer)
    scheduler_name = scheduler.__class__.__name__
    tr: TrainingResult = {
        "timestamp": timestamp,
        "model_name": "SimpleCNN",
        "dataset_name": "FashionMNIST" if ds.fashion_mnist else "MNIST",
        "optimizer_name": optimizer_name,
        "optimizer_args": optimizer_args,
        "scheduler_name": scheduler_name,  # TODO
        "batch_size": batch_size,
        "target_accuracy": target_accuracy,
        "acc_measure_freq": acc_measure_freq,
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "steps_to_target": len(train_losses),
    }

    log_batch_size = int(math.log2(batch_size))

    suffix = f"{optimizer_name}_b{log_batch_size}_{tr['dataset_name'][0].lower()}_{timestamp}"

    if save_model:
        model_path = RESULTS_PATH / f"model_{suffix}.pt"
        t.save(model, model_path)

    if save_tr:
        tr_path = RESULTS_PATH / f"tr_{suffix}.json"
        with open(tr_path, "w", encoding="utf-8") as f:
            json.dump(tr, f)

    return model, tr
