import abc
from typing import List, Tuple

import torch
from torch import nn

from region_graph import RegionNode

COMPUTE_LAYERS = ["cp"]


class ComputeLayer(nn.Module, abc.ABC):
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_in_components: int,
        num_out_components: int,
        **kwargs
    ):
        super().__init__()
        self.rg_nodes = rg_nodes
        self.num_in_components = num_in_components
        self.num_out_components = num_out_components


class MonotonicComputeLayer(ComputeLayer, abc.ABC):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class BornComputeLayer(ComputeLayer, abc.ABC):
    def forward(self, x: torch.Tensor, square: bool = False) -> torch.Tensor:
        raise NotImplementedError()
