import abc
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import distributions as ds
from torch import nn

from pcs.initializers import init_params_
from pcs.utils import (log_binomial, ohe, retrieve_complex_default_dtype,
                       retrieve_default_dtype, safelog)
from region_graph import RegionNode
from splines.bsplines import (basis_polyint, basis_polyval,
                              integrate_cartesian_basis,
                              splines_uniform_polynomial)


class InputLayer(nn.Module, abc.ABC):
    def __init__(self, rg_nodes: List[RegionNode], num_units: int, **kwargs):
        super().__init__()
        self.rg_nodes = rg_nodes
        self.num_units = num_units
        replica_indices = set(n.get_replica_idx() for n in rg_nodes)
        self.num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(self.num_replicas)
        ), "Replica indices should be consecutive, starting with 0."
        self.num_variables = len(set(v for n in rg_nodes for v in n.scope))

    @abc.abstractmethod
    def log_pf(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass


class MonotonicInputLayer(InputLayer, abc.ABC):
    @abc.abstractmethod
    def log_pf(self) -> torch.Tensor:
        pass


class BornInputLayer(InputLayer, abc.ABC):
    @abc.abstractmethod
    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class MonotonicEmbeddings(MonotonicInputLayer):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        num_states: int = 2,
        init_method: str = "dirichlet",
        init_scale: float = 1.0,
    ):
        super().__init__(rg_nodes, num_units)
        self.num_states = num_states
        weight = torch.empty(
            self.num_variables, self.num_replicas, self.num_units, num_states
        )
        init_params_(weight, init_method, init_scale=init_scale)
        self.weight = nn.Parameter(torch.log(weight), requires_grad=True)
        self._ohe = num_states <= 256

    def log_pf(self) -> torch.Tensor:
        # log_z: (1, num_vars, num_replicas, num_units)
        log_z = torch.logsumexp(self.weight, dim=-1).unsqueeze(dim=0)
        return log_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars)
        # self.weight: (num_vars, num_comps, num_states)
        if self._ohe:
            x = ohe(x, self.num_states)  # (-1, num_vars, num_states)
            # log_y: (-1, num_vars, num_replicas, num_units)
            log_y = torch.einsum("bvd,vrid->bvri", x, self.weight)
        else:
            weight = self.weight.permute(0, 3, 1, 2)
            log_y = weight[torch.arange(weight.shape[0], device=x.device), x]
        return log_y


class BornEmbeddings(BornInputLayer):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        num_states: int = 2,
        init_method: str = "normal",
        init_scale: float = 1.0,
        complex: bool = False,
        exp_reparam: bool = False,
        l2norm_reparam: bool = False,
    ):
        assert (
            not exp_reparam or not l2norm_reparam
        ), "Only one between 'exp_reparam' and 'l2norm_reparam' can be set true"
        super().__init__(rg_nodes, num_units)
        self.num_states = num_states
        complex_dtype = retrieve_complex_default_dtype()
        weight = torch.empty(
            self.num_variables,
            self.num_replicas,
            self.num_units,
            num_states,
            dtype=complex_dtype if complex else None,
        )
        init_params_(weight, init_method, init_scale=init_scale)
        if exp_reparam:
            weight = torch.log(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.complex = complex
        self.exp_reparam = exp_reparam
        self.l2norm_reparam = l2norm_reparam
        self._complex_dtype = complex_dtype
        self._ohe = num_states <= 256

    def _forward_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.exp_reparam:
            weight = torch.exp(self.weight)
        elif self.l2norm_reparam:
            weight = self.weight / torch.linalg.vector_norm(
                self.weight, ord=2, dim=2, keepdim=True
            )
        else:
            weight = self.weight
        if self.complex:
            # note: .conj() returns a view
            return weight, weight.conj()
        return weight, weight

    def log_pf(self) -> torch.Tensor:
        # weight: (num_vars, num_comps, num_states)
        # Get the weight and the conjugate weight tensors
        weight, weight_conj = self._forward_weight()
        z = torch.einsum(
            "vrid,vrjd->vrij", weight, weight_conj
        )  # (num_variables, num_replicas, num_units, num_units)
        if not self.complex:
            z = z.to(self._complex_dtype)
        z = torch.log(z)
        return z.unsqueeze(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars)
        # weight: (num_vars, num_comps, num_states)
        # Get the weight and the conjugate weight tensors
        weight, weight_conj = self._forward_weight()
        if self._ohe:
            ohe_dtype = self._complex_dtype if self.complex else None
            x = ohe(x, self.num_states, dtype=ohe_dtype)  # (-1, num_vars, num_states)
            # (-1, num_vars, num_replicas, num_units)
            y = torch.einsum("bvd,vrid->bvri", x, weight)
        else:
            weight = weight.permute(0, 3, 1, 2)
            y = weight[torch.arange(weight.shape[0], device=x.device), x]
        if not self.complex:
            y = y.to(self._complex_dtype)
        y = torch.log(y)
        return y


class MonotonicBinaryEmbeddings(MonotonicEmbeddings):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        init_method: str = "dirichlet",
        init_scale: float = 1.0,
    ):
        super().__init__(
            rg_nodes,
            num_units=num_units,
            num_states=2,
            init_method=init_method,
            init_scale=init_scale,
        )


class BornBinaryEmbeddings(BornEmbeddings):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        init_method: str = "normal",
        init_scale: float = 1.0,
        complex: bool = False,
        exp_reparam: bool = False,
    ):
        super().__init__(
            rg_nodes,
            num_units=num_units,
            num_states=2,
            init_method=init_method,
            init_scale=init_scale,
            complex=complex,
            exp_reparam=exp_reparam,
        )


class MonotonicBinomial(MonotonicInputLayer):
    def __init__(self, rg_nodes: List[RegionNode], num_units: int, num_states: int = 2):
        super().__init__(rg_nodes, num_units)
        self.num_states = num_states
        self.total_count = num_states - 1
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_units)
        init_params_(weight, "uniform", init_loc=0.0, init_scale=1.0)
        self.weight = nn.Parameter(torch.logit(weight), requires_grad=True)
        self.register_buffer(
            "log_bcoeffs",
            torch.tensor(
                [log_binomial(self.total_count, k) for k in range(num_states)],
                dtype=torch.get_default_dtype(),
            ),
        )
        self._log_sigmoid = nn.LogSigmoid()

    def log_pf(self) -> torch.Tensor:
        return torch.zeros(
            1,
            self.num_variables,
            self.num_replicas,
            self.num_units,
            device=self.weight.device,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars, 1, 1)
        # self.weight: (num_vars, num_units, num_states)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)

        # log_coeffs: (-1, num_vars, 1, 1)
        log_bcoeffs = self.log_bcoeffs[x]

        # log_success: (-1, num_vars, num_replicas, num_units)
        # log_failure: (-1, num_vars, num_replicas, num_units)
        log_probs = self._log_sigmoid(self.weight).unsqueeze(dim=0)
        log_success = x * log_probs
        log_failure = (self.total_count - x) * torch.log1p(-torch.exp(log_probs))

        # log_y: (-1, num_vars, num_replicas, num_units)
        log_y = log_bcoeffs + log_success + log_failure
        return log_y


class BornBinomial(BornInputLayer):
    def __init__(self, rg_nodes: List[RegionNode], num_units: int, num_states: int = 2):
        super().__init__(rg_nodes, num_units)
        self.num_states = num_states
        self.total_count = num_states - 1
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_units)
        init_params_(weight, "uniform", init_loc=0.0, init_scale=1.0)
        self.weight = nn.Parameter(torch.logit(weight), requires_grad=True)
        self.register_buffer(
            "log_bcoeffs",
            torch.tensor(
                [log_binomial(self.total_count, k) for k in range(num_states)],
                dtype=torch.get_default_dtype(),
            ),
        )
        self._log_sigmoid = nn.LogSigmoid()
        self._complex_dtype = retrieve_complex_default_dtype()

    def log_pf(self) -> torch.Tensor:
        # counts, log_bcoeefs: (num_states, 1, 1, 1)
        counts = (
            torch.arange(self.num_states, device=self.weight.device)
            .unsqueeze(dim=1)
            .unsqueeze(dim=2)
            .unsqueeze(dim=3)
        )
        log_bcoeffs = (
            self.log_bcoeffs.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
        )

        # log_success: (num_states, num_vars, num_replicas, num_units)
        # log_failure: (num_states, num_vars, num_replicas, num_units)
        log_success_probs = self._log_sigmoid(self.weight).unsqueeze(dim=0)
        log_success = counts * log_success_probs
        log_failure = (self.total_count - counts) * torch.log1p(
            -torch.exp(log_success_probs)
        )
        log_probs = log_bcoeffs + log_success + log_failure

        # log_z: (1, num_variables, num_replicas, num_units, num_units)
        m_lp, _ = torch.max(
            log_probs, dim=0, keepdim=True
        )  # (1, num_vars, num_replicas, num_units)
        e_lp = torch.exp(
            log_probs - m_lp
        )  # (num_states, num_vars, num_replicas, num_units)
        z = torch.einsum("dvri,dvrj->vrij", e_lp, e_lp)
        log_z = (
            torch.log(z.unsqueeze(dim=0))
            + m_lp.unsqueeze(dim=-2)
            + m_lp.unsqueeze(dim=-1)
        )
        return log_z.to(self._complex_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars, 1, 1)
        # self.weight: (num_vars, num_units)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)

        # log_coeffs: (-1, num_vars, 1, 1)
        log_bcoeffs = self.log_bcoeffs[x]

        # log_success: (-1, num_vars, num_units)
        # log_failure: (-1, num_vars, num_units)
        log_success_probs = self._log_sigmoid(self.weight).unsqueeze(dim=0)
        log_success = x * log_success_probs
        log_failure = (self.total_count - x) * torch.log1p(
            -torch.exp(log_success_probs)
        )

        # log_probs: (-1, num_vars, num_replicas, num_units)
        log_probs = log_bcoeffs + log_success + log_failure
        return log_probs.to(self._complex_dtype)


class NormalDistribution(MonotonicInputLayer):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        init_scale: float = 1.0,
    ):
        super().__init__(rg_nodes, num_units)
        # Initialize mean
        mu = torch.empty(1, self.num_variables, self.num_replicas, self.num_units)
        init_params_(mu, "normal", init_scale=1.0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # Initialize diagonal covariance matrix
        log_sigma = torch.empty(
            1, self.num_variables, self.num_replicas, self.num_units
        )
        init_params_(log_sigma, "normal", init_loc=0.0, init_scale=init_scale)
        self.log_sigma = nn.Parameter(log_sigma)
        self.eps = 1e-5

    def log_pf(self) -> torch.Tensor:
        return torch.zeros(
            1,
            self.num_variables,
            self.num_replicas,
            self.num_units,
            device=self.mu.device,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables) -> (-1, num_variables, 1, 1)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        scale = torch.exp(self.log_sigma) + self.eps
        # log_prob: (-1, num_vars, num_replicas, num_units)
        log_prob = ds.Normal(loc=self.mu, scale=scale).log_prob(x)
        return log_prob


class BornNormalDistribution(BornInputLayer):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        init_scale: float = 1.0,
    ):
        super().__init__(rg_nodes, num_units)
        self.log_two_pi = np.log(2.0 * np.pi)
        # Initialize mean
        mu = torch.empty(1, self.num_variables, self.num_replicas, self.num_units)
        init_params_(mu, "normal", init_scale=1.0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # Initialize diagonal covariance matrix
        log_sigma = torch.empty(
            1, self.num_variables, self.num_replicas, self.num_units
        )
        init_params_(log_sigma, "normal", init_loc=0.0, init_scale=init_scale)
        self.log_sigma = nn.Parameter(log_sigma, requires_grad=True)
        self.eps = 1e-5
        self._complex_dtype = retrieve_complex_default_dtype()

    def log_pf(self) -> torch.Tensor:
        mu = self.mu
        log_sigma = self.log_sigma
        # log_sum_cov, sq_norm_dist: (1, num_variables, num_replicas, num_units, num_units)
        log_cov = 2.0 * torch.log(torch.exp(log_sigma) + self.eps)
        m_s, _ = torch.max(log_cov, dim=-1, keepdim=True)
        cov = torch.exp(log_cov - m_s)
        log_sum_cov = m_s.unsqueeze(dim=-1) + torch.log(
            cov.unsqueeze(dim=-2) + cov.unsqueeze(dim=-1)
        )
        log_sq_dif_mu = 2.0 * safelog(
            torch.abs(mu.unsqueeze(dim=-2) - mu.unsqueeze(dim=-1))
        )
        sq_norm_dist = torch.exp(log_sq_dif_mu - log_sum_cov)
        # log_z: (1, num_variables, num_replicas, num_units, num_units)
        log_z = -0.5 * (self.log_two_pi + log_sum_cov + sq_norm_dist)
        return log_z.to(self._complex_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables) -> (-1, num_variables, 1, 1)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        scale = torch.exp(self.log_sigma) + self.eps
        # log_prob: (-1, num_variables, num_replicas, num_units)
        log_prob = ds.Normal(loc=self.mu, scale=scale).log_prob(x)
        return log_prob.to(self._complex_dtype)


class MonotonicBSplines(MonotonicInputLayer):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        order: int = 2,
        interval: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(rg_nodes, num_units)
        self.order = order
        self.interval = interval

        # Construct the basis functions (as polynomials) and the knots
        intervals, polynomials = splines_uniform_polynomial(
            order, num_units, interval=interval
        )
        numpy_dtype = retrieve_default_dtype(numpy=True)
        intervals = intervals.astype(numpy_dtype, copy=False)
        polynomials = polynomials.astype(numpy_dtype, copy=False)
        self.register_buffer("intervals", torch.from_numpy(intervals))
        self.register_buffer("polynomials", torch.from_numpy(polynomials))
        self.register_buffer(
            "_integral_basis", basis_polyint(self.intervals, self.polynomials)
        )

    def log_pf(self) -> torch.Tensor:
        sint = self._integral_basis
        # log_z: (1, 1, 1, num_units)
        log_z = safelog(sint)
        log_z = log_z.unsqueeze(dim=0)
        log_z = log_z.unsqueeze(dim=1)
        log_z = log_z.unsqueeze(dim=2)
        return log_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables)
        y = basis_polyval(
            self.intervals, self.polynomials, x
        )  # (-1, num_variables, num_units)
        return safelog(y).unsqueeze(dim=2)


class BornBSplines(BornInputLayer):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_units: int,
        order: int = 2,
        interval: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(rg_nodes, num_units)
        self.order = order
        self.interval = interval

        # Construct the basis functions (as polynomials) and the knots
        intervals, polynomials = splines_uniform_polynomial(
            order, num_units, interval=interval
        )
        numpy_dtype = retrieve_default_dtype(numpy=True)
        intervals = intervals.astype(numpy_dtype, copy=False)
        polynomials = polynomials.astype(numpy_dtype, copy=False)
        self.register_buffer("intervals", torch.from_numpy(intervals))
        self.register_buffer("polynomials", torch.from_numpy(polynomials))
        self.register_buffer(
            "_integral_cartesian_basis",
            integrate_cartesian_basis(self.intervals, self.polynomials),
        )
        self._complex_dtype = retrieve_complex_default_dtype()

    def log_pf(self) -> torch.Tensor:
        # z: (num_knots, num_knots)
        z = self._integral_cartesian_basis
        # log_z: (1, 1, 1, num_units, num_units)
        log_z = safelog(z)
        log_z = log_z.unsqueeze(dim=0)
        log_z = log_z.unsqueeze(dim=1)
        log_z = log_z.unsqueeze(dim=2)
        return log_z.to(self._complex_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables)
        y = basis_polyval(
            self.intervals, self.polynomials, x
        )  # (-1, num_variables, num_units)
        # log_y: (-1, num_variables, 1, num_units)
        log_y = safelog(y).unsqueeze(dim=2)
        return log_y.to(self._complex_dtype)
