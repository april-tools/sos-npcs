from typing import Iterable, Optional

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.parameters import (
    ConjugateParameter,
    OuterProductParameter,
    Parameter,
    ReduceProductParameter,
    ReduceSumParameter,
)
from cirkit.utils.scope import Scope
from layers import ConstantLayer, EmbeddingLayer


def integrate_embedding_layer(
    sl: EmbeddingLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    if sl.scope != scope:
        raise NotImplementedError()
    reduce_sum = ReduceSumParameter(sl.weight.shape, axis=3)
    reduce_prod1 = ReduceProductParameter(reduce_sum.shape, axis=2)
    reduce_prod2 = ReduceProductParameter(reduce_prod1.shape, axis=0)
    constant_value = Parameter.from_sequence(
        sl.weight.ref(), reduce_sum, reduce_prod1, reduce_prod2
    )
    sl = ConstantLayer(
        sl.scope, sl.num_output_units, sl.num_channels, value=constant_value
    )
    return CircuitBlock.from_layer(sl)


def multiply_embedding_layers(sl1: EmbeddingLayer, sl2: EmbeddingLayer) -> CircuitBlock:
    assert sl1.num_variables == sl2.num_variables
    assert sl1.num_channels == sl2.num_channels
    assert sl1.num_states == sl2.num_states
    sl_weight = Parameter.from_binary(
        OuterProductParameter(sl1.weight.shape, sl2.weight.shape, axis=1),
        sl1.weight.ref(),
        sl2.weight.ref(),
    )
    sl = EmbeddingLayer(
        sl1.scope | sl2.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_channels=sl1.num_channels,
        num_states=sl1.num_states,
        weight=sl_weight,
    )
    return CircuitBlock.from_layer(sl)


def conjugate_embedding_layer(sl: EmbeddingLayer) -> CircuitBlock:
    weight = Parameter.from_unary(ConjugateParameter(sl.weight.shape), sl.weight.ref())
    sl = EmbeddingLayer(
        sl.scope,
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_states=sl.num_states,
        weight=weight,
    )
    return CircuitBlock.from_layer(sl)
