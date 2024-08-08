from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.parameters.nodes import (
    TorchEntrywiseParameterOp,
    TorchParameterOp,
    TorchUnaryParameterOp,
)
from cirkit.symbolic.parameters import EntrywiseParameterOp


class DoubleClampParameter(EntrywiseParameterOp):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        *,
        eps: Optional[float] = None,
    ) -> None:
        super().__init__(in_shape)
        self.eps = eps

    @property
    def config(self) -> Dict[str, Any]:
        config = dict()
        if self.eps is not None:
            config.update(eps=self.eps)
        return config


class TorchDoubleClampParameter(TorchEntrywiseParameterOp):
    def __init__(
        self, in_shape: Tuple[int, ...], num_folds: int = 1, eps: Optional[float] = None
    ):
        super().__init__(in_shape, num_folds=num_folds)
        if eps is None:
            eps = torch.finfo(torch.get_default_dtype()).tiny
        self.eps = eps

    @property
    def config(self) -> Dict[str, Any]:
        return {"eps": self.eps}

    @torch.no_grad()
    @torch.compile()
    def forward(self, x: Tensor) -> Tensor:
        close_zero_mask = (x.real > -self.eps) & (x.real < self.eps)
        clamped_x = self.eps * (1.0 - 2.0 * torch.signbit(x.real))
        torch.where(close_zero_mask, clamped_x, x.real, out=x.real)
        return x


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


def compile_double_clamp_parameter(
    compiler: TorchCompiler, p: DoubleClampParameter
) -> TorchDoubleClampParameter:
    (in_shape,) = p.in_shapes
    return TorchDoubleClampParameter(in_shape, eps=p.eps)
