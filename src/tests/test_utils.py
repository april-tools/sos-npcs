import itertools

import numpy as np
import pytest


def generate_all_nary_samples(num_variables: int, arity: int = 2) -> np.ndarray:
    vs = list(range(arity))
    return np.asarray(list(itertools.product(vs, repeat=num_variables)))


def generate_all_binary_samples(num_variables: int) -> np.ndarray:
    return generate_all_nary_samples(num_variables, arity=2)


def generate_all_ternary_samples(num_variables: int) -> np.ndarray:
    return generate_all_nary_samples(num_variables, arity=3)


@pytest.mark.parametrize("num_variables,arity", list(itertools.product([1, 5], [2, 3])))
def test_generate_all_nary_samples(num_variables, arity):
    x = generate_all_nary_samples(num_variables, arity=arity)
    assert len(x) == int(arity**num_variables)
    assert np.all(np.isin(x, list(range(arity))))
