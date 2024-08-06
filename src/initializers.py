import functools
from typing import Any, Dict

from torch import Tensor, nn

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.initializers import InitializerFunc
from cirkit.symbolic.initializers import Initializer


class ExpUniformInitializer(Initializer):
    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        if a >= b:
            raise ValueError("The minimum should be strictly less than the maximum")
        self.a = a
        self.b = b

    @property
    def config(self) -> Dict[str, Any]:
        return dict(a=self.a, b=self.b)


def exp_uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    nn.init.uniform_(tensor, a=a, b=b)
    tensor.log_()
    return tensor


def compile_exp_uniform_initializer(
    compiler: TorchCompiler, init: ExpUniformInitializer
) -> InitializerFunc:
    return functools.partial(exp_uniform_, a=init.a, b=init.b)
