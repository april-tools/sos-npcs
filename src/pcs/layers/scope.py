import abc
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from pcs.utils import retrieve_complex_default_dtype, retrieve_default_dtype
from region_graph import RegionNode


class ScopeLayer(nn.Module, abc.ABC):
    def __init__(self, rg_nodes: List[RegionNode], dtype: Optional[torch.dtype] = None):
        super().__init__()
        scope = ScopeLayer.__build_scope(rg_nodes)
        if dtype is None:
            dtype = retrieve_default_dtype()
        self.register_buffer('scope', torch.from_numpy(scope).to(dtype))

    @staticmethod
    def __build_scope(rg_nodes: List[RegionNode]) -> np.ndarray:
        replica_indices = set(n.get_replica_idx() for n in rg_nodes)
        num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(num_replicas)
        ), "Replica indices should be consecutive, starting with 0."
        num_variables = len(set(v for n in rg_nodes for v in n.scope))
        scope = np.zeros(shape=(len(rg_nodes), num_variables, num_replicas), dtype=np.float64)
        for i, n in enumerate(rg_nodes):
            scope[i, list(n.scope), n.get_replica_idx()] = 1.0
        return scope


class MonotonicScopeLayer(ScopeLayer):
    def __init__(self, rg_nodes: List[RegionNode]):
        super().__init__(rg_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars, num_replicas, num_components)
        # y: (-1, num_folds, num_components)
        return torch.einsum('bvri,fvr->bfi', x, self.scope)


class BornScopeLayer(ScopeLayer):
    def __init__(self, rg_nodes: List[RegionNode]):
        super().__init__(rg_nodes, dtype=retrieve_complex_default_dtype())

    def forward(self, x: torch.Tensor, square: bool = False) -> torch.Tensor:
        if square:
            # x: (-1, num_vars, num_replicas, num_components, num_components)
            # y: (-1, num_folds, num_components, num_components)
            y = torch.einsum('bvrij,fvr->bfij', x, self.scope)
            return y

        # x: (-1, num_vars, num_replicas, num_components)
        # y: (-1, num_folds, num_components)
        y = torch.einsum('bvri,fvr->bfi', x, self.scope)
        return y
