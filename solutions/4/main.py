import argparse
import math
import os
import random
from typing import Literal, NamedTuple

import torch as t
from torch.optim import SGD, Adam


from src.heavy_ball import HeavyBall
from src.simple_cnn import SimpleCNN, DatasetSimpleCNN
from src.training import train, RESULTS_PATH

Optimizer = SGD | HeavyBall | Adam


class Args(NamedTuple):
    fashion_mnist: bool
    optimizer: Literal["SGD", "HeavyBall", "Adam", "Adam-m"]


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashion_mnist", "-f", action="store_true")
    parser.add_argument(
        "--optimizer",
        "-o",
        choices=["SGD", "HeavyBall", "Adam", "Adam-m"],
        required=True,
    )
    return Args(**vars(parser.parse_args()))


BATCH_SIZES: list[int] = (2 ** t.arange(4, 14)).tolist()
SEED = 42


def init_model_and_optimizer(args: Args) -> tuple[SimpleCNN, Optimizer]:
    model = SimpleCNN()
    lr = 1e-4
    match args.optimizer:
        case "SGD":
            optimizer = SGD(model.parameters(), lr)
        case "HeavyBall":
            momentum = 0.8
            optimizer = HeavyBall(model.parameters(), lr, momentum)
        case "Adam":
            betas = (0.9, 0.999)
            optimizer = Adam(model.parameters(), lr, betas)
        case "Adam-m":
            betas = (0, 0)
            optimizer = Adam(model.parameters(), lr, betas)
    return model, optimizer


def result_exists(optimizer_name: str, batch_size: int, fashion_mnist: bool) -> bool:
    log_batch_size_str = f"_b{int(math.log2(batch_size))}_"
    ds_char = "_f_" if fashion_mnist else "_m_"

    def _is_result(fname: str) -> bool:
        return (
            optimizer_name in fname and log_batch_size_str in fname and ds_char in fname
        )

    model_result_exists = any(
        fname.startswith("model_") and _is_result(fname)
        for fname in os.listdir(RESULTS_PATH)
    )
    th_result_exists = any(
        fname.startswith("th_") and _is_result(fname)
        for fname in os.listdir(RESULTS_PATH)
    )
    return model_result_exists and th_result_exists


def main() -> None:
    args = parse_args()
    print(f"{args = }")
    ds = DatasetSimpleCNN.load(fashion_mnist=args.fashion_mnist)
    for batch_size in BATCH_SIZES:
        if result_exists(args.optimizer, batch_size, fashion_mnist=args.fashion_mnist):
            print(f"{batch_size = }: result exists")
        else:
            print(f"{batch_size = }: starting training")
            model, optimizer = init_model_and_optimizer(args)
            th = train(model, ds, optimizer, batch_size)
            print()


if __name__ == "__main__":
    main()
