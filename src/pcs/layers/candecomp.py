from typing import List, Tuple

import torch
from torch import nn

from pcs.initializers import init_params_
from pcs.layers import BornComputeLayer, MonotonicComputeLayer
from pcs.utils import retrieve_complex_default_dtype


class MonotonicCPLayer(MonotonicComputeLayer):
    def __init__(
        self,
        num_folds: int,
        num_in_components: int,
        num_out_components: int,
        init_method: str = "dirichlet",
        init_scale: float = 1.0,
    ):
        super().__init__(num_folds, num_in_components, num_out_components)
        weight = torch.empty(num_folds, num_out_components, num_in_components)
        init_params_(weight, init_method, init_scale=init_scale)
        self.weight = nn.Parameter(torch.log(weight), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_folds, arity, num_in_components)
        # Compute the element-wise product
        x = torch.sum(x, dim=-2)  # (-1, num_folds, num_in_components)

        # Log-einsum-exp trick
        w = torch.exp(self.weight)
        m_x, _ = torch.max(x, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        e_x = torch.exp(x - m_x)  # (-1, num_folds, num_in_components)
        y = torch.einsum("fkp,bfp->bfk", w, e_x)  # (-1, num_folds, num_out_components)
        y = m_x + torch.log(y)  # (-1, num_folds, num_out_components)
        return y


class BornCPLayer(BornComputeLayer):
    def __init__(
        self,
        num_folds: int,
        num_in_components: int,
        num_out_components: int,
        init_method: str = "normal",
        init_scale: float = 1.0,
        complex: bool = False,
        exp_reparam: bool = False,
    ):
        super().__init__(num_folds, num_in_components, num_out_components)
        complex_dtype = retrieve_complex_default_dtype()
        weight = torch.empty(
            num_folds,
            num_out_components,
            num_in_components,
            dtype=complex_dtype if complex else None,
        )
        init_params_(weight, init_method, init_scale=init_scale)
        if exp_reparam:
            weight = torch.log(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.complex = complex
        self.exp_reparam = exp_reparam
        self._complex_dtype = complex_dtype

    def _forward_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = torch.exp(self.weight) if self.exp_reparam else self.weight
        if self.complex:
            # note: .conj() returns a view
            return weight, weight.conj()
        weight = weight.to(self._complex_dtype)
        return weight, weight

    def forward(self, x: torch.Tensor, square: bool = False) -> torch.Tensor:
        # Get the weight and the conjugate weight tensors
        weight, weight_conj = self._forward_weight()

        if square:
            # x: (-1, num_folds, num_in_components, num_in_components)
            # Compute the element-wise product
            x = torch.sum(
                x, dim=-3
            )  # (-1, num_folds, num_in_components, num_in_components)

            # Complex log-einsum-exp trick
            m_x, _ = torch.max(
                x.real, dim=-2, keepdim=True
            )  # (-1, num_folds, 1, num_in_components)
            e_x = torch.exp(
                x - m_x
            )  # (-1, num_folds, num_in_components, num_in_components)
            # x: (-1, num_folds, num_out_components, num_in_components)
            x = torch.einsum("fki,bfij->bfkj", weight, e_x)
            x = m_x + torch.log(x)
            m_x, _ = torch.max(
                x.real, dim=-1, keepdim=True
            )  # (-1, num_folds, num_out_components, 1)
            e_x = torch.exp(
                x - m_x
            )  # (-1, num_folds, num_out_components, num_in_components)
            # (-1, num_folds, num_out_components, num_out_components)
            x = torch.einsum("bfkj,flj->bfkl", e_x, weight_conj)
            x = m_x + torch.log(x)
            return x

        # x: (-1, num_folds, arity, num_in_components)
        # Compute the element-wise product
        x = torch.sum(x, dim=-2)  # (-1, num_folds, num_in_components)

        # Complex log-einsum-exp trick
        m_x, _ = torch.max(x.real, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        e_x = torch.exp(x - m_x)  # (-1, num_folds, num_in_components)
        # y: (-1, num_folds, num_out_components)
        y = torch.einsum("fkp,bfp->bfk", weight, e_x)
        y = m_x + torch.log(y)
        return y
