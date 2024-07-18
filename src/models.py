from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Sequence, cast, Iterator, Dict, Any

import numpy as np
import torch
from cirkit.backend.torch.circuits import TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.layers import TorchSumLayer, TorchLayer
from cirkit.backend.torch.optimization.layers import DenseKroneckerPattern
from cirkit.pipeline import PipelineContext, compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.initializers import NormalInitializer, UniformInitializer
from cirkit.symbolic.layers import (
    GaussianLayer,
    HadamardLayer,
    DenseLayer,
    CategoricalLayer,
)
from cirkit.symbolic.parameters import (
    ExpParameter,
    Parameter,
    ClampParameter,
    TensorParameter,
)
from cirkit.templates.region_graph import (
    RegionGraph,
    RandomBinaryTree,
    LinearRegionGraph,
)
from cirkit.utils.scope import Scope
from torch import Tensor, nn

import cirkit.symbolic.functional as SF

from layers import apply_dense_product


class PC(nn.Module, ABC):
    def __init__(self, num_variables: int) -> None:
        assert num_variables > 1
        super().__init__()
        self.num_variables = num_variables
        self.__cache_log_z: Optional[Tensor] = None

    def train(self, mode: bool = True):
        if mode:
            self.__cache_log_z = None
        else:
            with torch.no_grad():
                self.__cache_log_z = self.log_partition()
        super().train(mode)

    def forward(self, x: Tensor) -> Tensor:
        return self.log_score(x)

    def log_likelihood(self, x: Tensor) -> Tensor:
        log_z = (
            self.log_partition() if self.__cache_log_z is None else self.__cache_log_z
        )
        log_score = self.log_score(x)
        return log_score - log_z

    @abstractmethod
    def layers(self) -> Iterator[TorchLayer]: ...

    @abstractmethod
    def sum_layers(self) -> Iterator[TorchSumLayer]: ...

    @abstractmethod
    def log_partition(self) -> Tensor: ...

    @abstractmethod
    def log_score(self, x: Tensor) -> Tensor: ...


class MPC(PC):
    def __init__(
        self,
        num_variables: int,
        *,
        num_input_units: int,
        num_sum_units: int,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        num_components: int = 1,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        seed: int = 42,
    ) -> None:
        assert num_components > 0
        super().__init__(num_variables)
        self._pipeline = PipelineContext(
            backend="torch", semiring="lse-sum", fold=True, optimize=True
        )
        self._circuit, self._int_circuit = self._build_circuits(
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            num_components=num_components,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )
        self.register_buffer(
            "_mixing_log_weight", -torch.log(torch.tensor(num_components))
        )

    def layers(self) -> Iterator[TorchLayer]:
        return iter(self._circuit.layers)

    def sum_layers(self) -> Iterator[TorchSumLayer]:
        return filter(lambda l: isinstance(l, TorchSumLayer), self._circuit.layers)

    def log_partition(self) -> Tensor:
        log_z = self._int_circuit()
        return torch.logsumexp(self._mixing_log_weight + log_z, dim=0)

    def log_score(self, x: Tensor) -> Tensor:
        log_score = self._circuit(x)
        return torch.logsumexp(self._mixing_log_weight + log_score, dim=1)

    def _build_circuits(
        self,
        num_input_units: int,
        num_sum_units: int,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        num_components: int = 1,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        seed: int = 42,
    ) -> Tuple[TorchCircuit, TorchConstantCircuit]:
        # Build the region graphs
        rgs = _build_region_graphs(
            region_graph,
            num_components,
            self.num_variables,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )

        # Build one symbolic circuit for each region graph
        symbolic_circuits = _build_monotonic_symbolic_circuits(
            rgs,
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
        )

        with self._pipeline:
            # Merge the symbolic circuits into a single one having multiple outputs
            symbolic_circuit = SF.merge(symbolic_circuits)

            # Integrate the circuits (by integrating the merged symbolic representation)
            symbolic_int_circuit = SF.integrate(symbolic_circuit)

            # Compile the symbolic circuits
            circuit = cast(TorchCircuit, compile(symbolic_circuit))
            int_circuit = cast(TorchConstantCircuit, compile(symbolic_int_circuit))

        return circuit, int_circuit


class SOS(PC):
    def __init__(
        self,
        num_variables: int,
        *,
        num_input_units: int,
        num_sum_units: int,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        num_squares: int = 1,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        complex: bool = False,
        seed: int = 42,
    ) -> None:
        assert num_squares > 0
        super().__init__(num_variables)
        self._pipeline = PipelineContext(
            backend="torch", semiring="complex-lse-sum", fold=True, optimize=True
        )
        # Use a different optimization rule for the dense-kronecker pattern
        self._pipeline._compiler._optimization_registry["layer_shatter"].add_rule(
            apply_dense_product, signature=DenseKroneckerPattern
        )
        self._circuit, self._int_sq_circuit = self._build_circuits(
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            num_squares=num_squares,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            complex=complex,
            seed=seed,
        )
        self.register_buffer(
            "_mixing_log_weight", -torch.log(torch.tensor(num_squares))
        )

    def layers(self) -> Iterator[TorchLayer]:
        return iter(self._circuit.layers)

    def sum_layers(self) -> Iterator[TorchSumLayer]:
        return filter(lambda l: isinstance(l, TorchSumLayer), self._circuit.layers)

    def log_partition(self) -> Tensor:
        log_z = self._int_sq_circuit().real
        return torch.logsumexp(self._mixing_log_weight + log_z, dim=0)

    def log_score(self, x: Tensor) -> Tensor:
        log_score = 2.0 * self._circuit(x).real
        return torch.logsumexp(self._mixing_log_weight + log_score, dim=1)

    def _build_circuits(
        self,
        num_input_units: int,
        num_sum_units: int,
        *,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        num_squares: int = 1,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        complex: bool = False,
        seed: int = 42,
    ) -> Tuple[TorchCircuit, TorchConstantCircuit]:
        # Build the region graphs
        rgs = _build_region_graphs(
            region_graph,
            num_squares,
            self.num_variables,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )

        # Build one symbolic circuit for each region graph
        symbolic_circuits = _build_non_monotonic_symbolic_circuits(
            rgs,
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            complex=complex,
        )

        with self._pipeline:
            # Merge the symbolic circuits into a single one having multiple outputs
            symbolic_circuit = SF.merge(symbolic_circuits)

            # Square each symbolic circuit and merge them into a single one having multiple outputs
            symbolic_sq_circuits = [
                (SF.multiply(SF.conjugate(sc), sc) if complex else SF.multiply(sc, sc))
                for sc in symbolic_circuits
            ]
            symbolic_sq_circuit = SF.merge(symbolic_sq_circuits)

            # Integrate the squared circuits (by integrating the merged symbolic representation)
            symbolic_int_sq_circuit = SF.integrate(symbolic_sq_circuit)

            # Compile the symbolic circuits
            circuit = cast(TorchCircuit, compile(symbolic_circuit))
            int_sq_circuit = cast(
                TorchConstantCircuit, compile(symbolic_int_sq_circuit)
            )

        return circuit, int_sq_circuit


def _build_region_graphs(
    name: str,
    k: int,
    num_variables: int,
    structured_decomposable: bool = False,
    seed: int = 42,
) -> Sequence[RegionGraph]:
    if name == "rnd-bt":
        return [
            _build_rnd_bt_region_graph(
                num_variables,
                seed=(seed if structured_decomposable else seed + i * 123),
            )
            for i in range(k)
        ]
    elif name == "rnd-lt":
        return [
            _build_lt_region_graph(
                num_variables,
                random=True,
                seed=(seed if structured_decomposable else seed + i * 123),
            )
            for i in range(k)
        ]
    elif name == "lt":
        return [_build_lt_region_graph(num_variables, random=False) for _ in range(k)]
    raise NotImplementedError()


def _build_rnd_bt_region_graph(num_variables: int, seed: int = 42) -> RegionGraph:
    max_depth = int(np.ceil(np.log2(num_variables)))
    return RandomBinaryTree(num_variables, depth=max_depth, seed=seed)


def _build_lt_region_graph(
    num_variables: int, random: bool = False, seed: int = 42
) -> RegionGraph:
    return LinearRegionGraph(num_variables, random=random, seed=seed)


def _build_monotonic_symbolic_circuits(
    region_graphs: Sequence[RegionGraph],
    num_input_units: int,
    num_sum_units: int,
    *,
    input_layer: str,
    input_layer_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Circuit]:
    if input_layer_kwargs is None:
        input_layer_kwargs = {}

    def categorical_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> CategoricalLayer:
        assert "num_categories" in input_layer_kwargs
        return CategoricalLayer(
            scope,
            num_units,
            num_channels,
            num_categories=input_layer_kwargs["num_categories"],
            logits_factory=lambda shape: Parameter.from_leaf(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1))
            ),
        )

    def gaussian_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> GaussianLayer:
        return GaussianLayer(
            scope,
            num_units,
            num_channels,
            mean_factory=lambda shape: Parameter.from_leaf(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
            ),
            stddev_factory=lambda shape: Parameter.from_sequence(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1)),
                ExpParameter(shape),
                ClampParameter(shape, vmin=1e-5),
            ),
        )

    def hadamard_layer_factory(
        scope: Scope, num_input_units: int, arity: int
    ) -> HadamardLayer:
        return HadamardLayer(scope, num_input_units, arity)

    def dense_layer_factory(
        scope: Scope, num_input_units: int, num_output_units: int
    ) -> DenseLayer:
        return DenseLayer(
            scope,
            num_input_units,
            num_output_units,
            weight_factory=lambda shape: Parameter.from_unary(
                ExpParameter(shape),
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1)),
            ),
        )

    def build_symbolic_circuit(rg: RegionGraph) -> Circuit:
        assert input_layer in ["categorical", "gaussian"]
        if input_layer == "categorical":
            input_factory = categorical_layer_factory
        elif input_layer == "gaussian":
            input_factory = gaussian_layer_factory
        else:
            raise NotImplementedError()
        return Circuit.from_region_graph(
            rg,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_factory=input_factory,
            sum_factory=dense_layer_factory,
            prod_factory=hadamard_layer_factory,
        )

    return list(map(lambda rg: build_symbolic_circuit(rg), region_graphs))


def _build_non_monotonic_symbolic_circuits(
    region_graphs: Sequence[RegionGraph],
    num_input_units: int,
    num_sum_units: int,
    *,
    input_layer: str,
    input_layer_kwargs: Optional[Dict[str, Any]] = None,
    complex: bool = False,
) -> List[Circuit]:
    if input_layer_kwargs is None:
        input_layer_kwargs = {}

    def categorical_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> CategoricalLayer:
        assert "num_categories" in input_layer_kwargs
        return CategoricalLayer(
            scope,
            num_units,
            num_channels,
            num_categories=input_layer_kwargs["num_categories"],
            logits_factory=lambda shape: Parameter.from_leaf(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1))
            ),
        )

    def gaussian_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> GaussianLayer:
        return GaussianLayer(
            scope,
            num_units,
            num_channels,
            mean_factory=lambda shape: Parameter.from_leaf(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
            ),
            stddev_factory=lambda shape: Parameter.from_sequence(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1)),
                ExpParameter(shape),
                ClampParameter(shape, vmin=1e-5),
            ),
        )

    def hadamard_layer_factory(
        scope: Scope, num_input_units: int, arity: int
    ) -> HadamardLayer:
        return HadamardLayer(scope, num_input_units, arity)

    def dense_layer_factory(
        scope: Scope, num_input_units: int, num_output_units: int
    ) -> DenseLayer:
        weight_dtype = DataType.COMPLEX if complex else DataType.REAL
        return DenseLayer(
            scope,
            num_input_units,
            num_output_units,
            weight_factory=lambda shape: Parameter.from_leaf(
                TensorParameter(
                    *shape, initializer=UniformInitializer(0.0, 1.0), dtype=weight_dtype
                )
            ),
        )

    def build_symbolic_circuit(rg: RegionGraph) -> Circuit:
        assert input_layer in ["categorical", "gaussian"]
        if input_layer == "categorical":
            input_factory = categorical_layer_factory
        elif input_layer == "gaussian":
            input_factory = gaussian_layer_factory
        else:
            raise NotImplementedError()
        return Circuit.from_region_graph(
            rg,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_factory=input_factory,
            sum_factory=dense_layer_factory,
            prod_factory=hadamard_layer_factory,
        )

    return list(map(lambda rg: build_symbolic_circuit(rg), region_graphs))
