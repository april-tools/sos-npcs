from collections.abc import Iterator

import torch
from torch import optim

OPTIMIZERS_NAMES = ["Adam", "Adagrad", "NAdam", "SGD"]


def setup_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    opt: str,
    learning_rate: float,
    decay1: float = 0.9,
    decay2: float = 0.999,
    momentum: float = 0.9,
    weight_decay: float = 0,
):
    if opt == "Adagrad":
        return optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    if opt == "Adam":
        return optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(decay1, decay2),
            weight_decay=weight_decay,
        )
    if opt == "NAdam":
        return optim.NAdam(
            parameters,
            lr=learning_rate,
            betas=(decay1, decay2),
            weight_decay=weight_decay,
        )
    if opt == "SGD":
        return optim.SGD(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimiser called {opt}")
