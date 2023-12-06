import random

import torch as t

from src import Dataset, CNN, TrainingHistory, train

LR = 1e-3
N_EPOCHS = 200
SEED = 42


def main() -> None:
    random.seed(SEED)
    t.manual_seed(SEED)
    model = CNN()
    ds = Dataset.load()
    optimizer = t.optim.AdamW(model.parameters(), lr=LR)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    th: TrainingHistory = train(
        model,
        ds,
        optimizer,
        scheduler=scheduler,
        n_epochs=N_EPOCHS,
    )


if __name__ == "__main__":
    main()
