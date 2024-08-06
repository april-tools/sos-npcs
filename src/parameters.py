from functools import cached_property
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.parameters.nodes import (
    TorchParameterOp,
    TorchUnaryParameterOp,
)
from cirkit.symbolic.parameters import ParameterOp, UnaryParameterOp

# class FlattenParameter(UnaryParameterOp):
#     def __init__(self, in_shape: Tuple[int, ...], *, start_axis: int = 0, end_axis: int = -1):
#         start_axis = start_axis if start_axis >= 0 else start_axis + len(in_shape)
#         assert 0 <= start_axis < len(in_shape)
#         end_axis = end_axis if end_axis >= 0 else end_axis + len(in_shape)
#         assert 0 <= end_axis < len(in_shape)
#         assert start_axis < end_axis
#         super().__init__(in_shape)
#         self.start_axis = start_axis
#         self.end_axis = end_axis
#
#     @cached_property
#     def shape(self) -> Tuple[int, ...]:
#         flattened_dim = np.prod([self.in_shapes[0][i] for i in range(self.start_axis, self.end_axis + 1)])
#         return *self.in_shapes[0][:self.start_axis], flattened_dim, *self.in_shapes[0][self.end_axis + 1:]
#
#     @property
#     def config(self) -> Dict[str, Any]:
#         return dict(start_axis=self.start_axis, end_axis=self.end_axis)


# class EinsumParameter(ParameterOp):
#     def __init__(self, *in_shapes: Tuple[int, ...], einsum: str):
#         super().__init__(*in_shapes)
#         self.einsum = einsum
#         self._output_shape = EinsumParameter._compute_output_shape(*in_shapes, einsum=einsum)
#
#     @staticmethod
#     def _compute_output_shape(*in_shapes: Tuple[int, ...], einsum: str) -> Tuple[int, ...]:
#         idx_to_dim: Dict[str, int] = {}
#         in_idx, out_idx = einsum.split('->')
#         for in_shape, multi_in_idx in zip(in_shapes, in_idx.split(',')):
#             for idx, einsum_idx in enumerate(multi_in_idx):
#                 if einsum_idx in idx_to_dim:
#                     if in_shape[idx] != idx_to_dim[einsum_idx]:
#                         raise ValueError(
#                             f"Einsum string shape mismatch, found {in_idx[idx]} but expected {idx_to_dim[einsum_idx]}"
#                         )
#                     continue
#                 idx_to_dim[einsum_idx] = in_shape[idx]
#         return tuple(idx_to_dim[einsum_idx] for einsum_idx in out_idx)
#
#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self._output_shape
#
#     @property
#     def config(self) -> Dict[str, Any]:
#         return {'einsum': self.einsum}


class TorchFlattenParameter(TorchUnaryParameterOp):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
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
    def config(self) -> Dict[str, Any]:
        return {"start_dim": self.start_dim, "end_dim": self.end_dim}

    @cached_property
    def shape(self) -> Tuple[int, ...]:
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
    def __init__(self, *in_shapes: Tuple[int, ...], num_folds: int = 1, einsum: str):
        if "f" in einsum:
            raise ValueError(
                "The einsum string should not contain the reserved index 'f'"
            )
        super().__init__(*in_shapes, num_folds=num_folds)
        self.einsum = einsum
        self._output_shape = TorchEinsumParameter._compute_output_shape(
            *in_shapes, einsum=einsum
        )
        in_idx, out_idx = einsum.split("->")
        self._processed_einsum = (
            ",".join("f" + multi_in_idx for multi_in_idx in in_idx.split(","))
            + "->"
            + ("f" + out_idx)
        )

    @staticmethod
    def _compute_output_shape(
        *in_shapes: Tuple[int, ...], einsum: str
    ) -> Tuple[int, ...]:
        idx_to_dim: Dict[str, int] = {}
        in_idx, out_idx = einsum.split("->")
        for in_shape, multi_in_idx in zip(in_shapes, in_idx.split(",")):
            for idx, einsum_idx in enumerate(multi_in_idx):
                if einsum_idx in idx_to_dim:
                    if in_shape[idx] != idx_to_dim[einsum_idx]:
                        raise ValueError(
                            f"Einsum string shape mismatch, found {in_idx[idx]} but expected {idx_to_dim[einsum_idx]}"
                        )
                    continue
                idx_to_dim[einsum_idx] = in_shape[idx]
        return tuple(idx_to_dim[einsum_idx] for einsum_idx in out_idx)

    @property
    def config(self) -> Dict[str, Any]:
        return {"einsum": self.einsum}

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._output_shape

    def forward(self, *xs: Tensor) -> Tensor:
        return torch.einsum(self._processed_einsum, *xs)
