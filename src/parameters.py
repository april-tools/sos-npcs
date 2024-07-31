from typing import Tuple, Dict, Any

import torch
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.parameters.nodes import TorchEntrywiseParameterOp
from cirkit.symbolic.parameters import EntrywiseParameterOp
from torch import Tensor


class ScaledSigmoidParameter(EntrywiseParameterOp):
    def __init__(self, in_shape: Tuple[int, ...], vmin: float, vmax: float, scale: float = 1.0):
        super().__init__(in_shape)
        self.vmin = vmin
        self.vmax = vmax
        self.scale = scale

    @property
    def config(self) -> Dict[str, Any]:
        return dict(vmin=self.vmin, vmax=self.vmax, scale=self.scale)


class TorchScaledSigmoidParameter(TorchEntrywiseParameterOp):
    def __init__(
        self, in_shape: Tuple[int, ...], *, vmin: float, vmax: float, scale: float, num_folds: int = 1
    ) -> None:
        super().__init__(in_shape, num_folds=num_folds)
        assert 0 <= vmin < vmax, "Must provide 0 <= vmin < vmax."
        assert scale > 0.0
        self.vmin = vmin
        self.vmax = vmax
        self.scale = scale

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(vmin=self.vmin, vmax=self.vmax, scale=self.scale)
        return config

    @torch.compile()
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x * self.scale) * (self.vmax - self.vmin) + self.vmin


def compile_scaled_sigmoid_parameter(
    compiler: TorchCompiler, p: ScaledSigmoidParameter
) -> TorchScaledSigmoidParameter:
    (in_shape,) = p.in_shapes
    return TorchScaledSigmoidParameter(in_shape, vmin=p.vmin, vmax=p.vmax, scale=p.scale)
