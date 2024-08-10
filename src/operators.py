from typing import Iterable, Optional

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import CategoricalLayer
from cirkit.symbolic.parameters import (
    ConjugateParameter,
    ExpParameter,
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
    assert tuple(sl.scope | scope) == tuple(scope)
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


def multiply_categorical_embedding_layers(
    sl1: CategoricalLayer, sl2: EmbeddingLayer
) -> CircuitBlock:
    assert sl1.num_variables == sl2.num_variables
    assert sl1.num_channels == sl2.num_channels
    assert sl1.num_categories == sl2.num_states

    if sl1.logits is None:
        sl_scores = sl1.probs.ref()
    else:
        sl_scores = Parameter.from_unary(
            ExpParameter(sl1.logits.shape), sl1.logits.ref()
        )

    sl_weight = Parameter.from_binary(
        OuterProductParameter(sl_scores.shape, sl2.weight.shape, axis=1),
        sl_scores,
        sl2.weight.ref(),
    )
    sl = EmbeddingLayer(
        sl1.scope | sl2.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_channels=sl1.num_channels,
        num_states=sl1.num_categories,
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
