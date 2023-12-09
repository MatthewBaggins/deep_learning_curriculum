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

from .heavy_ball import HeavyBall
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
    optimizer_name: OptimizerName
    optimizer_args: dict[str, float | tuple[float, float]]
    scheduler_args: dict[str, float]
    # numbers
    batch_size: int
    target_accuracy: float
    acc_measure_freq: int
    train_losses: list[float]
    test_accuracies: list[float]
    steps_to_target: int


OptimizerName = Literal["SGD", "HeavyBall", "Adam", "Adam-m"]


def get_optimizer_name_and_args(
    optimizer: SGD | HeavyBall | Adam,
) -> tuple[OptimizerName, dict[str, float | tuple[float, float]]]:
    assert isinstance(
        optimizer, (SGD, HeavyBall, Adam)
    ), f"Invalid optimizer: type={optimizer.__class__.__name__}"
    if isinstance(optimizer, SGD):
        optimizer_name = "SGD"
        optimizer_args: dict[str, float] = {"lr": optimizer.param_groups[0]["lr"]}
    elif isinstance(optimizer, HeavyBall):
        optimizer_name = "HeavyBall"
        optimizer_args = {"lr": optimizer.lr, "momentum": optimizer.momentum}
    else:  # isinstance(optimizer, Adam):
        if optimizer.param_groups[0]["betas"] == (0, 0):
            optimizer_name = "Adam-m"
        else:
            optimizer_name = "Adam"
        optimizer_args: dict[str, float | tuple[float, float]] = {
            "lr": optimizer.param_groups[0]["lr"],
            "betas": optimizer.param_groups[0]["betas"],
        }
    return optimizer_name, optimizer_args


# Target accuract on the val/test set after which training terminates
TARGET_ACCURACY = 0.97
# Maximum number of steps (batches) after which training terminates
MAX_N_STEPS = 2**14
# How often to measure accuracy, each 100 steps (batches) by default
ACC_MEASURE_FREQ = 100

RESULTS_PATH = Path()
while not "src" in os.listdir(RESULTS_PATH):
    RESULTS_PATH = ".." / RESULTS_PATH

RESULTS_PATH /= "results"
if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir()


def train(
    model: SimpleCNN,
    ds: DatasetSimpleCNN,
    optimizer: SGD | HeavyBall | Adam,
    batch_size: int,
    *,
    target_accuracy: float = TARGET_ACCURACY,
    acc_measure_freq: int = ACC_MEASURE_FREQ,
    max_n_steps: int = MAX_N_STEPS,
    verbose: bool = True,
    scheduler: lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau | None = None,
) -> tuple[SimpleCNN, TrainingResult]:
    loss_fn = t.nn.CrossEntropyLoss()

    train_x_batches = split_into_batches(ds.train_x, batch_size)  # .tolist()
    train_y_batches = split_into_batches(ds.train_y, batch_size)  # .tolist()
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
            running_loss = t.tensor(train_losses[-acc_measure_freq:]).mean().item()
            if verbose:
                print(f"Step {step_i}: {running_loss=:.4f}, {test_acc=:.2%}")
            if test_acc >= target_accuracy:
                print(f"{target_accuracy=} achieved after {step_i} steps")
            if isinstance(scheduler, lr_scheduler.LRScheduler):
                scheduler.step()
            elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss)

    timestamp = datetime.now().isoformat("T", "minutes").replace(":", "")
    optimizer_name, optimizer_args = get_optimizer_name_and_args(optimizer)
    tr: TrainingResult = {
        "timestamp": timestamp,
        "model_name": "SimpleCNN",
        "dataset_name": "FashionMNIST" if ds.fashion_mnist else "MNIST",
        "optimizer_name": optimizer_name,
        "optimizer_args": optimizer_args,
        "scheduler_args": {},  # TODO
        "batch_size": batch_size,
        "target_accuracy": target_accuracy,
        "acc_measure_freq": acc_measure_freq,
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "steps_to_target": len(train_losses),
    }

    log_batch_size = int(math.log2(batch_size))

    suffix = f"{optimizer_name}_b{log_batch_size}_{tr['dataset_name'][0].lower()}_{timestamp}"

    model_filename = f"model_{suffix}.pt"
    t.save(model, RESULTS_PATH / model_filename)

    tr_filename = f"tr_{suffix}.json"
    with open(RESULTS_PATH / tr_filename, "w", encoding="utf-8") as f:
        json.dump(tr, f)

    return model, tr
