import itertools
from functools import cached_property
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cirkit.backend.torch.parameters.nodes import (
    TorchParameterOp,
    TorchUnaryParameterOp,
)


class TorchFlattenParameter(TorchUnaryParameterOp):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        num_folds: int = 1,
        start_dim: int = 0,
        end_dim: int = -1,
    ):
        super().__init__(in_shape, num_folds=num_folds)
        start_dim = start_dim if start_dim >= 0 else start_dim + len(in_shape)
        assert 0 <= start_dim < len(in_shape)
        end_dim = end_dim if end_dim >= 0 else end_dim + len(in_shape)
        assert 0 <= end_dim < len(in_shape)
        assert start_dim < end_dim
        self.start_dim = start_dim
        self.end_dim = end_dim

    @property
    def config(self) -> dict[str, Any]:
        return {
            "in_shape": self.in_shape,
            "start_dim": self.start_dim,
            "end_dim": self.end_dim,
        }

    @cached_property
    def shape(self) -> tuple[int, ...]:
        flattened_dim = np.prod(
            [self.in_shapes[0][i] for i in range(self.start_dim, self.end_dim + 1)]
        )
        return (
            *self.in_shapes[0][: self.start_dim],
            flattened_dim,
            *self.in_shapes[0][self.end_dim + 1 :],
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, start_dim=self.start_dim + 1, end_dim=self.end_dim + 1)


class TorchEinsumParameter(TorchParameterOp):
    def __init__(
        self,
        in_shapes: tuple[tuple[int, ...], ...],
        einsum: tuple[tuple[int, ...], ...],
        num_folds: int = 1,
    ):
        if len(in_shapes) != len(einsum) - 1:
            raise ValueError("Number of inputs and einsum shapes mismatch")
        idx_to_dim: dict[int, int] = {}
        for in_shape, multi_in_idx in zip(in_shapes, einsum):
            for i, einsum_idx in enumerate(multi_in_idx):
                if einsum_idx not in idx_to_dim:
                    idx_to_dim[einsum_idx] = in_shape[i]
                    continue
                if in_shape[i] != idx_to_dim[einsum_idx]:
                    raise ValueError(
                        f"Einsum shape mismatch, found {in_shape[i]} "
                        f"but expected {idx_to_dim[einsum_idx]}"
                    )
                continue
        super().__init__(*in_shapes, num_folds=num_folds)
        # Pre-compute the output shape of the einsum
        self._output_shape = tuple(
            idx_to_dim[einsum_idx] + 1 for einsum_idx in einsum[-1]
        )
        # Add fold dimension in both inputs and outputs of the einsum
        self.einsum = tuple(
            (0,) + tuple(map(lambda i: i + 1, einsum_idx)) for einsum_idx in einsum
        )

    @property
    def config(self) -> dict[str, Any]:
        unfolded_einsum = tuple(idx[1:] for idx in self.einsum)
        return {"in_shapes": self.in_shapes, "einsum": unfolded_einsum}

    @property
    def shape(self) -> tuple[int, ...]:
        return self._output_shape

    def forward(self, *xs: Tensor) -> Tensor:
        einsum_args = tuple(itertools.chain.from_iterable(zip(xs, self.einsum[:-1])))
        return torch.einsum(*einsum_args, self.einsum[-1])
