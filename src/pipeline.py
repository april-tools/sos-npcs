from typing import cast

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.optimization.registry import ParameterOptRegistry
from cirkit.pipeline import PipelineContext
from initializers import compile_exp_uniform_initializer
from optimizations import OuterProductReduceSumPattern, apply_prod_sum_einsum


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
    ctx.add_initializer_compilation_rule(compile_exp_uniform_initializer)
    compiler = cast(TorchCompiler, ctx._compiler)
    parameter_opt_registry = cast(
        ParameterOptRegistry, compiler._optimization_registry["parameter"]
    )
    parameter_opt_registry.add_rule(
        apply_prod_sum_einsum, signature=OuterProductReduceSumPattern
    )
    return ctx
