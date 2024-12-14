import itertools
from typing import cast

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
    def entries(cls) -> list[type[TorchParameterNode]]:
        return [TorchReduceSumParameter, TorchOuterProductParameter]


def apply_prod_sum_einsum(
    compiler: TorchCompiler, match: ParameterOptMatch
) -> tuple[TorchEinsumParameter] | tuple[TorchEinsumParameter, TorchFlattenParameter]:
    outer_prod = cast(TorchOuterProductParameter, match.entries[1])
    reduce_sum = cast(TorchReduceSumParameter, match.entries[0])
    in_shape1, in_shape2 = outer_prod.in_shapes
    if len(in_shape1) > 4:
        raise NotImplementedError()
    outer_dim = outer_prod.dim
    reduce_dim = reduce_sum.dim

    # in_idx1 = [0, 1, 2, ..., N - 1]
    in_idx1: tuple[int, ...] = tuple(range(len(in_shape1)))
    # in_idx2 = [0, 1, 2, ..., N + 1, ..., N - 1]
    in_idx2: tuple[int, ...] = (
        tuple(range(outer_dim))
        + (len(in_shape1),)
        + tuple(range(outer_dim + 1, len(in_shape1)))
    )
    # Apply the reduction to the indices, as to get the output indices of the einsum
    reduce_idx: list[tuple[int, ...]] = (
        list((i,) for i in range(outer_dim))
        + [(outer_dim, len(in_shape1))]
        + list((i,) for i in range(outer_dim + 1, len(in_shape1)))
    )
    del reduce_idx[reduce_dim]
    out_idx: tuple[int, ...] = tuple(itertools.chain.from_iterable(reduce_idx))

    # If we are reducing the dimension along which we compute the Kronecker product,
    # we just need an einsum
    einsum = TorchEinsumParameter(
        outer_prod.in_shapes, einsum=(in_idx1, in_idx2, out_idx)
    )
    if outer_dim == reduce_dim:
        return (einsum,)

    # If we are NOT reducing the dimension along which we compute the Kronecker product,
    # we need to flatten some dimensions after the einsum
    if reduce_dim < outer_dim:
        start_dim, end_dim = outer_dim - 1, outer_dim
    else:
        start_dim, end_dim = outer_dim, outer_dim + 1
    flatten = TorchFlattenParameter(einsum.shape, start_dim=start_dim, end_dim=end_dim)
    return einsum, flatten
