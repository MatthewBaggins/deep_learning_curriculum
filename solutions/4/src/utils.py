import torch as t


def split_into_batches(
    x: t.Tensor, batch_size: int, *, strict: bool = False
) -> t.Tensor:
    assert x.ndim >= 1
    if strict:
        assert len(x) % batch_size == 0
    n_batches = len(x) // batch_size
    x_batched = [x[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]
    return t.stack(x_batched)
