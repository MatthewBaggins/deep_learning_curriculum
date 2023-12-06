import os
from pathlib import Path
import random

import torch as t

from . import CNN, TrainingHistory


def load_model(path: Path) -> CNN:
    assert path.exists()
    assert str(path).endswith(".pt")
    with open(path, "rb") as f:
        model = t.load(f)
    assert isinstance(model, CNN)
    return model


def load_latest() -> tuple[CNN, TrainingHistory]:
    filenames = sorted(os.listdir("results"))
    latest_model_path = "results" / Path(
        next(
            fname
            for fname in filenames
            if fname.startswith("model") and fname.endswith(".pt")
        )
    )
    latest_th_path = "results" / Path(
        next(
            fname
            for fname in filenames
            if fname.startswith("th") and fname.endswith(".json")
        )
    )
    model = load_model(latest_model_path)
    th = TrainingHistory.load_from_json(latest_th_path)
    return model, th


def seed(val: float | None) -> None:
    random.seed(val)
    t.manual_seed(val)
