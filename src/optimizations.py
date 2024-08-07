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

    in_idx1 = list(range(len(in_shape1)))  # [0, 1, 2, ..., N - 1]
    in_idx2 = in_idx1.copy()  # [0, 1, 2, ..., K, ..., N - 1]
    in_idx2[outer_dim] = len(in_shape1)  # [0, 1, 2, ..., N, ..., N - 1]
    if outer_dim == reduce_dim:
        out_idx2 = list(range(len(in_shape1)))  # [0, 1, 2, ..., K, ..., N - 1]
        del out_idx2[reduce_dim]  # [0, 1, 2, ..., K - 1, K + 1, ..., N - 1]
        flatten_start_dim = None
        flatten_end_dim = None
    else:
        # out_idx2: [0, 1, 2, ..., K, N, K + 1, ..., N - 1]
        out_idx2 = (
            list(range(outer_dim + 1))
            + [len(in_shape1)]
            + list(range(outer_dim + 1, len(in_shape1)))
        )
        if reduce_dim < outer_dim:
            del out_idx2[
                reduce_dim
            ]  # [0, 1, 2, ..., J - 1, J + 1, ..., K, N, K + 1, ..., N - 1]
            flatten_start_dim = outer_dim - 1
            flatten_end_dim = outer_dim
        else:
            del out_idx2[
                reduce_dim + 1
            ]  # [0, 1, 2, ..., K, N, K + 1, ..., J - 1, J + 1, ..., N - 1]
            flatten_start_dim = outer_dim
            flatten_end_dim = outer_dim + 1

    # TODO: refactor TorchEinsumParameter to not accept only strings
    indices = ["r", "s", "t", "u", "v", "w", "x", "y", "z"]
    in_idx1 = "".join([indices[i] for i in in_idx1])
    in_idx2 = "".join([indices[i] for i in in_idx2])
    out_idx = "".join([indices[i] for i in out_idx2])
    einsum = TorchEinsumParameter(
        *outer_prod.in_shapes, einsum=f"{in_idx1},{in_idx2}->{out_idx}"
    )
    if flatten_start_dim is None:
        assert flatten_end_dim is None
        return (einsum,)
    assert flatten_end_dim is not None
    flatten = TorchFlattenParameter(
        einsum.shape, start_dim=flatten_start_dim, end_dim=flatten_end_dim
    )
    return einsum, flatten
