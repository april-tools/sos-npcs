from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring, SumProductSemiring, ComplexLSESumSemiring
from cirkit.backend.torch.utils import csafelog
from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.layers import InputLayer
from cirkit.symbolic.parameters import Parameter, ParameterFactory, TensorParameter
from cirkit.utils.scope import Scope


class EmbeddingLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        num_states: int = 2,
        weight: Optional[Parameter] = None,
        weight_factory: Optional[ParameterFactory] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        self.num_states = num_states
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_leaf(
                    TensorParameter(*self.weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self.weight_shape)
        if weight.shape != self.weight_shape:
            raise ValueError(
                f"Expected parameter shape {self.weight_shape}, found {weight.shape}"
            )
        self.weight = weight

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(num_states=self.num_states)
        return config

    @property
    def weight_shape(self) -> Tuple[int, ...]:
        return (
            self.num_variables,
            self.num_output_units,
            self.num_channels,
            self.num_states,
        )

    @property
    def params(self) -> Dict[str, Parameter]:
        return {"weight": self.weight}


class ConstantLayer(InputLayer):
    def __init__(
        self, scope: Scope, num_output_units: int, num_channels: int, value: Parameter
    ):
        super().__init__(scope, num_output_units, num_channels)
        if value.shape != self.value_shape:
            raise ValueError(
                f"Expected parameter shape {self.value_shape}, found {value.shape}"
            )
        self.value = value

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def params(self) -> Dict[str, Parameter]:
        params = super().params
        params.update(value=self.value)
        return params


class TorchEmbeddingLayer(TorchInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        num_states: int = 2,
        weight: Optional[TorchParameter] = None,
        semiring: Optional[Semiring] = None,
    ) -> None:
        if num_states <= 0:
            raise ValueError("The number of states for Embedding must be positive")
        if semiring != ComplexLSESumSemiring:
            raise NotImplementedError("The Embedding layer is implemented to work with the complex-lse-sum semiring")
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.num_folds = num_folds
        self.num_states = num_states
        if not self._valid_parameter_shape(weight):
            raise ValueError(
                f"The number of folds and shape of 'weight' must match the layer's"
            )
        self.weight = weight

    def _valid_parameter_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return p.shape == (
            len(self.scope),
            self.num_output_units,
            self.num_channels,
            self.num_states,
        )

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(num_states=self.num_states)
        return config

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return {"weight": self.weight}

    def forward(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Embedding should be discrete
        x = F.one_hot(x, self.num_states)  # (F, C, B, D, num_states)
        weight = self.weight()
        x = x.to(weight.dtype)
        x = torch.einsum("fcbdi,fdkci->fbkc", x, weight)
        x = torch.prod(x, dim=-1)  # (F, B, K)
        return csafelog(self.semiring.cast(x))


class TorchConstantLayer(TorchInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        value: TorchParameter,
        semiring: Optional[Semiring] = None,
    ) -> None:
        assert value.num_folds == num_folds
        assert value.shape == (num_output_units,)
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.value = value

    @property
    def params(self) -> Dict[str, TorchParameter]:
        params = super().params
        params.update(value=self.value)
        return params

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        value = self.value().unsqueeze(dim=1)  # (F, 1, Ko)
        # (F, Ko) -> (F, B, O)
        value = value.expand(value.shape[0], x.shape[2], value.shape[2])
        return self.semiring.map_from(value, SumProductSemiring)


def compile_embedding_layer(
    compiler: TorchCompiler, sl: EmbeddingLayer
) -> TorchEmbeddingLayer:
    weight = compiler.compile_parameter(sl.weight)
    return TorchEmbeddingLayer(
        sl.scope,
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_states=sl.num_states,
        weight=weight,
        semiring=compiler.semiring,
    )


def compile_constant_layer(
    compiler: TorchCompiler, sl: ConstantLayer
) -> TorchConstantLayer:
    value = compiler.compile_parameter(sl.value)
    return TorchConstantLayer(
        sl.scope,
        sl.num_output_units,
        num_channels=sl.num_channels,
        value=value,
        semiring=compiler.semiring,
    )
