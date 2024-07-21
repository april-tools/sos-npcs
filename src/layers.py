from typing import Optional, Dict, Any, Tuple, cast

import torch
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import TorchSumLayer, TorchDenseLayer
from cirkit.backend.torch.optimization.registry import LayerOptMatch
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring


class TorchDenseProductLayer(TorchSumLayer):
    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        num_folds: int = 1,
        weight1: TorchParameter,
        weight2: TorchParameter,
        semiring: Optional[Semiring] = None,
    ) -> None:
        assert num_input_units == weight1.shape[1] * weight2.shape[1]
        assert num_output_units == weight1.shape[0] * weight2.shape[0]
        assert weight1.num_folds == num_folds
        assert weight2.num_folds == num_folds
        super().__init__(
            num_input_units,
            num_output_units,
            arity=1,
            num_folds=num_folds,
            semiring=semiring,
        )
        self._in_shape = (weight1.shape[1], weight2.shape[1])
        self.weight1 = weight1
        self.weight2 = weight2

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "num_folds": self.num_folds,
        }

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return dict(weight1=self.weight1, weight2=self.weight2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (F, H=1, B, Ki) -> (F, B, Ki)
        x = x.squeeze(dim=1)
        # x: (F, B, Ki) -> (F, B, Kj, Kk)
        x = x.view(x.shape[0], x.shape[1], *self._in_shape)
        # weight1: (F, Kp, Kj)
        weight1 = self.weight1()
        # weight2: (F, Kq, Kk)
        weight2 = self.weight2()
        # y: (F, B, Kp, Kk)
        y = self.semiring.einsum(
            "fbjk,fpj->fbpk", inputs=(x,), operands=(weight1,), dim=-2, keepdim=True
        )
        # y: (F, B, Kp, Kq)
        y = self.semiring.einsum(
            "fbpk,fqk->fbpq", inputs=(y,), operands=(weight2,), dim=-1, keepdim=True
        )
        # return y: (F, B, Kp * Kq) = (F, B, Ko)
        return y.view(y.shape[0], y.shape[1], -1)


def apply_dense_product(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> Tuple[TorchDenseProductLayer]:
    # Retrieve the matched dense layer and the inputs to the kronecker parameter node
    dense = cast(TorchDenseLayer, match.entries[0])
    weight_patterns = match.pentries[0]["weight"]
    kronecker = weight_patterns[0].entries[0]
    weight1_output, weight2_output = dense.weight.node_inputs(kronecker)

    # Build new torch parameter computational graphs by taking
    # the sub-computational graph rooted at the inputs of the kronecker parameter node
    weight1, weight2 = dense.weight.extract_subgraphs(weight1_output, weight2_output)

    # Instantiate two torch dense product layer
    dprod = TorchDenseProductLayer(
        dense.num_input_units,
        dense.num_output_units,
        weight1=weight1,
        weight2=weight2,
        semiring=compiler.semiring,
    )
    return (dprod,)
