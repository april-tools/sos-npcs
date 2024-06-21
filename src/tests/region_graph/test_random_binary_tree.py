import itertools
import math

import pytest

from region_graph.random_binary_tree import RandomBinaryTree
from tests.region_graph.test_region_graph import (
    check_region_graph_save_load, check_region_partition_layers)


@pytest.mark.parametrize(
    "num_variables,num_repetitions",
    itertools.product([9, 13, 16], [1, 4]),
)
def test_random_binary_tree(num_variables: int, num_repetitions: int) -> None:
    rg = RandomBinaryTree(num_variables, num_repetitions=num_repetitions)
    assert rg.num_variables == num_variables
    assert rg.is_smooth
    assert rg.is_decomposable
    if num_repetitions == 1:
        assert rg.is_structured_decomposable
    check_region_partition_layers(rg, bottom_up=True)
    check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)
