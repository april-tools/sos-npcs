import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch import Tensor, nn

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.layers import TorchInnerLayer, TorchInputLayer, TorchLayer
from cirkit.pipeline import compile
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.initializers import NormalInitializer, UniformInitializer
from cirkit.symbolic.layers import (
    CategoricalLayer,
    GaussianLayer,
    EmbeddingLayer,
)
from cirkit.symbolic.parameters import (
    ClampParameter,
    ExpParameter,
    LogSoftmaxParameter,
    Parameter,
    ScaledSigmoidParameter,
    TensorParameter,
    SoftmaxParameter,
)
from cirkit.templates.region_graph import (
    RegionGraph,
    LinearTree,
    QuadTree,
    RandomBinaryTree,
)
from cirkit.utils.scope import Scope
from initializers import ExpUniformInitializer
from pipeline import setup_pipeline_context


class PC(nn.Module, ABC):
    def __init__(
        self, num_variables: int, image_shape: Optional[Tuple[int, int, int]] = None
    ) -> None:
        assert num_variables > 1
        if image_shape is not None:
            assert np.prod(image_shape) == num_variables
        super().__init__()
        self.num_variables = num_variables
        self.image_shape = image_shape
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

    def num_params(self, requires_grad: bool = True) -> int:
        return self.num_input_params(requires_grad) + self.num_sum_params(requires_grad)

    def num_input_params(self, requires_grad: bool = True) -> int:
        params = itertools.chain(*[l.parameters() for l in self.input_layers()])
        if requires_grad:
            params = filter(lambda p: p.requires_grad, params)
        num_params = sum(
            (2 * p.numel()) if p.is_complex() else p.numel() for p in params
        )
        return num_params

    def num_sum_params(self, requires_grad: bool = True) -> int:
        params = itertools.chain(*[l.parameters() for l in self.inner_layers()])
        if requires_grad:
            params = filter(lambda p: p.requires_grad, params)
        num_params = sum(
            (2 * p.numel()) if p.is_complex() else p.numel() for p in params
        )
        return num_params

    @abstractmethod
    def layers(self) -> Iterator[TorchLayer]:
        ...

    @abstractmethod
    def input_layers(self) -> Iterator[TorchInputLayer]:
        ...

    @abstractmethod
    def inner_layers(self) -> Iterator[TorchInnerLayer]:
        ...

    @abstractmethod
    def log_partition(self) -> Tensor:
        ...

    @abstractmethod
    def log_score(self, x: Tensor) -> Tensor:
        ...


class MPC(PC):
    def __init__(
        self,
        num_variables: int,
        image_shape: Optional[Tuple[int, int, int]] = None,
        *,
        num_input_units: int,
        num_sum_units: int,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        num_components: int = 1,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        mono_clamp: bool = False,
        seed: int = 42,
    ) -> None:
        assert num_components > 0
        super().__init__(num_variables, image_shape)
        self._pipeline = setup_pipeline_context(semiring="lse-sum")
        self._circuit, self._int_circuit = self._build_circuits(
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            num_components=num_components,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            mono_clamp=mono_clamp,
            seed=seed,
        )
        self.register_buffer(
            "_mixing_log_weight", -torch.log(torch.tensor(num_components))
        )

    def layers(self) -> Iterator[TorchLayer]:
        return iter(self._circuit.layers)

    def input_layers(self) -> Iterator[TorchInnerLayer]:
        return filter(lambda l: isinstance(l, TorchInputLayer), self._circuit.layers)

    def inner_layers(self) -> Iterator[TorchInnerLayer]:
        return filter(lambda l: isinstance(l, TorchInnerLayer), self._circuit.layers)

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
        mono_clamp: bool = False,
        seed: int = 42,
    ) -> Tuple[TorchCircuit, TorchConstantCircuit]:
        # Build the region graphs
        rgs = _build_region_graphs(
            region_graph,
            num_components,
            num_variables=self.num_variables,
            image_shape=self.image_shape,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )

        # Build one symbolic circuit for each region graph
        num_channels = 1 if self.image_shape is None else self.image_shape[0]
        sym_circuits = _build_monotonic_sym_circuits(
            rgs,
            num_channels,
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            mono_clamp=mono_clamp,
        )

        with self._pipeline:
            # Merge the symbolic circuits into a single one having multiple outputs
            sym_circuit = SF.concatenate(sym_circuits)

            # Integrate the circuits (by integrating the merged symbolic representation)
            sym_int_circuit = SF.integrate(sym_circuit)

            # Compile the symbolic circuits
            circuit = cast(TorchCircuit, compile(sym_circuit))
            int_circuit = cast(TorchConstantCircuit, compile(sym_int_circuit))

        return circuit, int_circuit


class SOS(PC):
    def __init__(
        self,
        num_variables: int,
        image_shape: Optional[Tuple[int, int, int]] = None,
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
        super().__init__(num_variables, image_shape)
        self._pipeline = setup_pipeline_context(semiring="complex-lse-sum")
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

    def input_layers(self) -> Iterator[TorchInnerLayer]:
        return filter(lambda l: isinstance(l, TorchInputLayer), self._circuit.layers)

    def inner_layers(self) -> Iterator[TorchInnerLayer]:
        return filter(lambda l: isinstance(l, TorchInnerLayer), self._circuit.layers)

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
        non_mono_clamp: bool = False,
        complex: bool = False,
        seed: int = 42,
    ) -> Tuple[TorchCircuit, TorchConstantCircuit]:
        # Build the region graphs
        rgs = _build_region_graphs(
            region_graph,
            num_squares,
            num_variables=self.num_variables,
            image_shape=self.image_shape,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )

        # Build one symbolic circuit for each region graph
        num_channels = 1 if self.image_shape is None else self.image_shape[0]
        sym_circuits = _build_non_monotonic_sym_circuits(
            rgs,
            num_channels,
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            complex=complex,
        )

        with self._pipeline:
            # Merge the symbolic circuits into a single one having multiple outputs
            sym_circuit = SF.concatenate(sym_circuits)

            # Square each symbolic circuit and merge them into a single one having multiple outputs
            sym_sq_circuits = [
                (SF.multiply(SF.conjugate(sc), sc) if complex else SF.multiply(sc, sc))
                for sc in sym_circuits
            ]
            sym_sq_circuit = SF.concatenate(sym_sq_circuits)

            # Integrate the squared circuits (by integrating the merged symbolic representation)
            sym_int_sq_circuit = SF.integrate(sym_sq_circuit)

            # Compile the symbolic circuits
            circuit = cast(TorchCircuit, compile(sym_circuit))
            int_sq_circuit = cast(TorchConstantCircuit, compile(sym_int_sq_circuit))

        return circuit, int_sq_circuit


class ExpSOS(PC):
    def __init__(
        self,
        num_variables: int,
        image_shape: Optional[Tuple[int, int, int]] = None,
        *,
        num_input_units: int,
        num_sum_units: int,
        mono_num_input_units: int = 2,
        mono_num_sum_units: int = 2,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        mono_clamp: bool = False,
        complex: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(num_variables, image_shape)
        self._pipeline = setup_pipeline_context(semiring="complex-lse-sum")
        # Introduce optimization rules
        self._circuit, self._mono_circuit, self._int_circuit = self._build_circuits(
            num_input_units,
            num_sum_units,
            mono_num_input_units,
            mono_num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            mono_clamp=mono_clamp,
            complex=complex,
            seed=seed,
        )

    def layers(self) -> Iterator[TorchLayer]:
        return itertools.chain(self._circuit.layers, self._mono_circuit.layers)

    def input_layers(self) -> Iterator[TorchInnerLayer]:
        return itertools.chain(
            filter(lambda l: isinstance(l, TorchInputLayer), self._circuit.layers),
            filter(lambda l: isinstance(l, TorchInputLayer), self._mono_circuit.layers),
        )

    def inner_layers(self) -> Iterator[TorchInnerLayer]:
        return itertools.chain(
            filter(lambda l: isinstance(l, TorchInnerLayer), self._circuit.layers),
            filter(lambda l: isinstance(l, TorchInnerLayer), self._mono_circuit.layers),
        )

    def log_partition(self) -> Tensor:
        return self._int_circuit().real

    def log_score(self, x: Tensor) -> Tensor:
        sq_log_score = 2.0 * self._circuit(x).real
        mono_log_score = self._mono_circuit(x).real
        return (sq_log_score + mono_log_score).squeeze(dim=1)

    def _build_circuits(
        self,
        num_input_units: int,
        num_sum_units: int,
        mono_num_input_units: int = 2,
        mono_num_sum_units: int = 2,
        *,
        input_layer: str,
        input_layer_kwargs: Optional[Dict[str, Any]] = None,
        region_graph: str = "rnd-bt",
        structured_decomposable: bool = False,
        mono_clamp: bool = False,
        complex: bool = False,
        seed: int = 42,
    ) -> Tuple[TorchCircuit, TorchCircuit, TorchConstantCircuit]:
        # Build the region graphs
        rgs = _build_region_graphs(
            region_graph,
            1,
            num_variables=self.num_variables,
            image_shape=self.image_shape,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )
        assert len(rgs) == 1

        # Build one symbolic circuit for each region graph
        num_channels = 1 if self.image_shape is None else self.image_shape[0]
        sym_circuits = _build_non_monotonic_sym_circuits(
            rgs,
            num_channels,
            num_input_units,
            num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            complex=complex,
        )

        sym_mono_circuits = _build_monotonic_sym_circuits(
            rgs,
            num_channels,
            mono_num_input_units,
            mono_num_sum_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            mono_clamp=mono_clamp,
        )
        assert len(sym_circuits) == 1
        assert len(sym_mono_circuits) == 1
        (sym_circuit,) = sym_circuits
        (sym_mono_circuit,) = sym_mono_circuits

        with self._pipeline:
            # Square the symbolic circuit and make the product with the monotonic circuit
            if complex:
                # Apply the conjugate operator if the circuit is complex
                sym_prod_circuit = SF.multiply(
                    SF.multiply(sym_mono_circuit, SF.conjugate(sym_circuit)),
                    sym_circuit,
                )
            else:
                sym_prod_circuit = SF.multiply(
                    SF.multiply(sym_mono_circuit, sym_circuit), sym_circuit
                )

            # Integrate the overall product circuit
            sym_int_circuit = SF.integrate(sym_prod_circuit)

            # Compile the symbolic circuits
            circuit = cast(TorchCircuit, compile(sym_circuit))
            mono_circuit = cast(TorchCircuit, compile(sym_mono_circuit))
            int_circuit = cast(TorchConstantCircuit, compile(sym_int_circuit))

        return circuit, mono_circuit, int_circuit


def _build_region_graphs(
    name: str,
    k: int,
    num_variables: Optional[int] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
    structured_decomposable: bool = False,
    seed: int = 42,
) -> Sequence[RegionGraph]:
    if name == "rnd-bt":
        assert num_variables is not None
        return [
            _build_rnd_bt_region_graph(
                num_variables,
                seed=(seed if structured_decomposable else seed + i * 123),
            )
            for i in range(k)
        ]
    elif name == "rnd-lt":
        assert num_variables is not None
        return [
            _build_lt_region_graph(
                num_variables,
                random=True,
                seed=(seed if structured_decomposable else seed + i * 123),
            )
            for i in range(k)
        ]
    elif name == "lt":
        assert num_variables is not None
        return [_build_lt_region_graph(num_variables, random=False) for _ in range(k)]
    elif name == "qt-2":
        assert image_shape is not None
        return [
            _build_qt_region_graph(image_shape, num_patch_splits=2) for _ in range(k)
        ]
    elif name in ["qt", "qt-4"]:
        assert image_shape is not None
        return [
            _build_qt_region_graph(image_shape, num_patch_splits=4) for _ in range(k)
        ]
    raise NotImplementedError()


def _build_rnd_bt_region_graph(num_variables: int, seed: int = 42) -> RegionGraph:
    max_depth = int(np.ceil(np.log2(num_variables)))
    return RandomBinaryTree(num_variables, depth=max_depth, seed=seed)


def _build_lt_region_graph(
    num_variables: int, random: bool = False, seed: int = 42
) -> RegionGraph:
    return LinearTree(num_variables, randomize=random, seed=seed)


def _build_qt_region_graph(
    image_shape: Tuple[int, int, int], num_patch_splits: int
) -> RegionGraph:
    num_channels, height, width = image_shape
    return QuadTree((height, width), num_patch_splits=num_patch_splits)


def _build_monotonic_sym_circuits(
    region_graphs: Sequence[RegionGraph],
    num_channels: int,
    num_input_units: int,
    num_sum_units: int,
    *,
    input_layer: str,
    input_layer_kwargs: Optional[Dict[str, Any]] = None,
    mono_clamp: bool = False,
) -> List[Circuit]:
    if input_layer_kwargs is None:
        input_layer_kwargs = {}

    def weight_factory_clamp(shape: Tuple[int, ...]) -> Parameter:
        return Parameter.from_unary(
            ClampParameter(shape, vmin=1e-19),
            TensorParameter(*shape, initializer=UniformInitializer(0.01, 0.99)),
        )

    def weight_factory_exp(shape: Tuple[int, ...]) -> Parameter:
        return Parameter.from_unary(
            ExpParameter(shape),
            TensorParameter(*shape, initializer=ExpUniformInitializer(0.0, 1.0)),
        )

    def categorical_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> CategoricalLayer:
        assert "num_categories" in input_layer_kwargs
        return CategoricalLayer(
            scope,
            num_units,
            num_channels,
            num_categories=input_layer_kwargs["num_categories"],
            logits_factory=lambda shape: Parameter.from_unary(
                LogSoftmaxParameter(shape),
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0)),
            ),
        )

    def embedding_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> EmbeddingLayer:
        assert "num_states" in input_layer_kwargs
        return EmbeddingLayer(
            scope,
            num_units,
            num_channels,
            num_states=input_layer_kwargs["num_states"],
            weight_factory=lambda shape: Parameter.from_unary(
                SoftmaxParameter(shape),
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0)),
            ),
        )

    def gaussian_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> GaussianLayer:
        return GaussianLayer(
            scope,
            num_units,
            num_channels,
            mean_factory=lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
            ),
            stddev_factory=lambda shape: Parameter.from_sequence(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0)),
                ScaledSigmoidParameter(shape, vmin=1e-5, vmax=1.0),
            ),
        )

    def build_sym_circuit(rg: RegionGraph) -> Circuit:
        assert input_layer in ["embedding", "categorical", "gaussian"]
        weight_factory = weight_factory_clamp if mono_clamp else weight_factory_exp
        if input_layer == "categorical":
            sum_product = "cp-t"
            input_factory = categorical_layer_factory
        elif input_layer == "gaussian":
            sum_product = "cp-t" if rg.num_variables == 2 else "cp"
            input_factory = gaussian_layer_factory
        else:
            sum_product = "cp-t"
            input_factory = embedding_layer_factory
        return Circuit.from_region_graph(
            rg,
            num_channels=num_channels,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_factory=input_factory,
            sum_product=sum_product,
            sum_weight_factory=weight_factory,
        )

    return list(map(lambda rg: build_sym_circuit(rg), region_graphs))


def _build_non_monotonic_sym_circuits(
    region_graphs: Sequence[RegionGraph],
    num_channels: int,
    num_input_units: int,
    num_sum_units: int,
    *,
    input_layer: str,
    input_layer_kwargs: Optional[Dict[str, Any]] = None,
    complex: bool = False,
) -> List[Circuit]:
    if input_layer_kwargs is None:
        input_layer_kwargs = {}

    def weight_factory(shape: Tuple[int, ...]) -> Parameter:
        weight_dtype = DataType.COMPLEX if complex else DataType.REAL
        return Parameter.from_input(
            TensorParameter(
                *shape, initializer=UniformInitializer(0.0, 1.0), dtype=weight_dtype
            )
        )

    def categorical_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> CategoricalLayer:
        assert "num_categories" in input_layer_kwargs
        return CategoricalLayer(
            scope,
            num_units,
            num_channels,
            num_categories=input_layer_kwargs["num_categories"],
            logits_factory=lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
            ),
        )

    def embedding_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> EmbeddingLayer:
        assert "num_states" in input_layer_kwargs
        return EmbeddingLayer(
            scope,
            num_units,
            num_channels,
            num_states=input_layer_kwargs["num_states"],
            weight_factory=weight_factory,
        )

    def gaussian_layer_factory(
        scope: Scope, num_units: int, num_channels: int
    ) -> GaussianLayer:
        return GaussianLayer(
            scope,
            num_units,
            num_channels,
            mean_factory=lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0))
            ),
            stddev_factory=lambda shape: Parameter.from_sequence(
                TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0)),
                ScaledSigmoidParameter(shape, vmin=1e-5, vmax=1.0),
            ),
        )

    def build_sym_circuit(rg: RegionGraph) -> Circuit:
        assert input_layer in ["categorical", "embedding", "gaussian"]
        if input_layer == "categorical":
            sum_product = "cp-t"
            input_factory = categorical_layer_factory
        elif input_layer == "gaussian":
            sum_product = "cp-t" if rg.num_variables == 2 else "cp"
            input_factory = gaussian_layer_factory
        else:
            sum_product = "cp-t"
            input_factory = embedding_layer_factory
        return Circuit.from_region_graph(
            rg,
            num_channels=num_channels,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_factory=input_factory,
            sum_product=sum_product,
            sum_weight_factory=weight_factory,
        )

    return list(map(lambda rg: build_sym_circuit(rg), region_graphs))
