from cirkit.pipeline import PipelineContext
from initializers import compile_exp_uniform_initializer


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
    return ctx
