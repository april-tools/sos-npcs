from typing import List, Tuple, Type, Union, cast

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.optimization.registry import (
    ParameterOptMatch,
    ParameterOptPatternDefn,
)
from cirkit.backend.torch.parameters.nodes import (
    TorchOuterProductParameter,
    TorchParameterNode,
    TorchReduceSumParameter,
)
from parameters import TorchEinsumParameter, TorchFlattenParameter


class OuterProductReduceSumPattern(ParameterOptPatternDefn):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchParameterNode]]:
        return [TorchReduceSumParameter, TorchOuterProductParameter]


def apply_prod_sum_einsum(
    compiler: TorchCompiler, match: ParameterOptMatch
) -> Union[
    Tuple[TorchEinsumParameter], Tuple[TorchEinsumParameter, TorchFlattenParameter]
]:
    outer_prod = cast(TorchOuterProductParameter, match.entries[1])
    reduce_sum = cast(TorchReduceSumParameter, match.entries[0])
    in_shape1, in_shape2 = outer_prod.in_shapes
    if len(in_shape1) > 4:
        raise NotImplementedError()
    outer_dim = outer_prod.dim
    reduce_dim = reduce_sum.dim

    cross_indices = ["j", "k"]
    indices = ["a", "b", "c", "d", "r", "s", "t", "u"]
    print(in_shape1, in_shape2, outer_dim)
    lhs_dim = len(in_shape1[:outer_dim])
    rhs_dim = len(in_shape1) - lhs_dim
    lhs_in_idx = "".join(indices[i] for i in range(lhs_dim))
    rhs_in_idx = "".join(indices[i] for i in range(lhs_dim, rhs_dim))
    in_idx = ",".join(
        [
            lhs_in_idx + cross_indices[0] + rhs_in_idx,
            lhs_in_idx + cross_indices[1] + rhs_in_idx,
        ]
    )
    if outer_dim == reduce_dim:
        out_idx = lhs_in_idx + rhs_in_idx
        einsum = TorchEinsumParameter(
            *outer_prod.in_shapes, einsum=in_idx + "->" + out_idx
        )
        return (einsum,)

    out_idx = lhs_in_idx + cross_indices[0] + cross_indices[1] + rhs_in_idx
    if reduce_dim < outer_dim:
        out_shape = (
            *in_shape1[:reduce_dim],
            *in_shape1[reduce_dim + 1 : outer_dim],
            in_shape1[outer_dim],
            in_shape2[outer_dim],
            *in_shape1[outer_dim + 1 :],
        )
        out_idx = out_idx[:reduce_dim] + out_idx[reduce_dim + 1 :]
        flatten = TorchFlattenParameter(
            out_shape, start_dim=len(lhs_in_idx) - 1, end_dim=len(lhs_in_idx)
        )
    else:
        out_shape = (
            *in_shape1[:outer_dim],
            in_shape1[outer_dim],
            in_shape2[outer_dim],
            *in_shape1[outer_dim + 1 : reduce_dim],
            *in_shape1[reduce_dim + 1 :],
        )
        out_idx = out_idx[: reduce_dim + 1] + out_idx[reduce_dim + 2 :]
        flatten = TorchFlattenParameter(
            out_shape, start_dim=len(lhs_in_idx), end_dim=len(lhs_in_idx) + 1
        )
    einsum = TorchEinsumParameter(*outer_prod.in_shapes, einsum=in_idx + "->" + out_idx)
    return einsum, flatten
