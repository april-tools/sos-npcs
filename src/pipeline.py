from typing import cast

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.optimization.registry import ParameterOptRegistry
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.layers import LayerOperation
from initializers import compile_exp_uniform_initializer
from layers import compile_constant_layer, compile_embedding_layer
from operators import (
    conjugate_embedding_layer,
    integrate_embedding_layer,
    multiply_embedding_layers,
)
from optimizations import OuterProductReduceSumPattern, apply_prod_sum_einsum
from parameters import compile_double_clamp_parameter


def setup_pipeline_context(
    *,
    backend: str = "torch",
    semiring: str = "lse-sum",
    fold: bool = True,
    optimize: bool = True,
) -> PipelineContext:
    ctx = PipelineContext(
        backend=backend, semiring=semiring, fold=fold, optimize=optimize
    )
    ctx.add_operator_rule(LayerOperation.INTEGRATION, integrate_embedding_layer)
    ctx.add_operator_rule(LayerOperation.MULTIPLICATION, multiply_embedding_layers)
    ctx.add_operator_rule(LayerOperation.CONJUGATION, conjugate_embedding_layer)
    ctx.add_parameter_compilation_rule(compile_double_clamp_parameter)
    ctx.add_initializer_compilation_rule(compile_exp_uniform_initializer)
    ctx.add_layer_compilation_rule(compile_embedding_layer)
    ctx.add_layer_compilation_rule(compile_constant_layer)
    compiler = cast(TorchCompiler, ctx._compiler)
    parameter_opt_registry = cast(
        ParameterOptRegistry, compiler._optimization_registry["parameter"]
    )
    parameter_opt_registry.add_rule(
        apply_prod_sum_einsum, signature=OuterProductReduceSumPattern
    )
    return ctx
