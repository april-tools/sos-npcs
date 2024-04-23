import numpy as np

from region_graph import RegionGraph, RegionNode, PartitionNode


def LinearTree(
        num_variables: int,
        num_repetitions: int = 1,
        randomize: bool = False,
        seed: int = 42,
        sd: bool = False
) -> RegionGraph:
    root = RegionNode(range(num_variables))
    rg = RegionGraph()
    rg.add_node(root)
    random_state = None

    for _ in range(num_repetitions):
        if randomize and (sd or random_state is None):
            # If structured-decomposable, then use the same sampled v-tree for each repetition
            random_state = np.random.RandomState(seed)
        parent_node = root
        vars = list(range(num_variables))
        if randomize:
            random_state.shuffle(vars)
        for i, v in enumerate(vars[:-1]):
            partition_node = PartitionNode(set(parent_node.scope))
            rg.add_edge(partition_node, parent_node)
            leaf_node = RegionNode({v})
            if i == num_variables - 2:
                rest_node = RegionNode({vars[-1]})
            else:
                rest_node = RegionNode({j for j in vars[i + 1:]})
            rg.add_edge(leaf_node, partition_node)
            rg.add_edge(rest_node, partition_node)
            parent_node = rest_node

    return rg
