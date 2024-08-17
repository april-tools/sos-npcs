import itertools
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from scipy import integrate

from models import MPC, PC, SOS, ExpSOS
from tests.test_utils import generate_all_binary_samples


def check_normalized_log_scores(model: PC, x: torch.Tensor) -> torch.Tensor:
    scores = model.log_score(x)
    assert scores.shape == (len(x), 1)
    assert torch.all(torch.isfinite(scores))
    assert torch.allclose(
        torch.logsumexp(scores, dim=0).exp(), torch.tensor(1.0), atol=1e-15
    )
    return scores


def check_evi_ll(model: PC, x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(dim=1)
    lls = model.log_likelihood(x)
    assert lls.shape == (len(x), 1)
    assert torch.all(torch.isfinite(lls))
    assert torch.allclose(
        torch.logsumexp(lls, dim=0).exp(), torch.tensor(1.0), atol=1e-15
    )
    return lls


def check_pdf(model, interval: Optional[Tuple[float, float]] = None):
    pdf = lambda y, x: torch.exp(model.log_likelihood(torch.Tensor([[[x, y]]])))
    if interval is None:
        a, b = -64.0, 64.0
    else:
        a, b = interval
    ig, err = integrate.dblquad(pdf, a, b, a, b)
    assert np.isclose(ig, 1.0, atol=1e-15)


@pytest.mark.parametrize(
    "num_variables,num_components,num_units,region_graph,sd",
    list(
        itertools.product([9, 12], [1, 4], [1, 3], ["rnd-bt", "rnd-lt"], [False, True])
    ),
)
def test_discrete_monotonic_pc(
    num_variables, num_components, num_units, region_graph, sd
):
    if region_graph == "qt":
        if num_variables == 9:
            image_shape = (1, 3, 3)
        else:  # num_variables == 12
            image_shape = (1, 4, 3)
    else:
        image_shape = None
    model = MPC(
        num_variables,
        image_shape=image_shape,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer="categorical",
        input_layer_kwargs={"num_categories": 2},
        num_components=num_components,
        region_graph=region_graph,
        structured_decomposable=sd,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "num_variables,num_squares,num_units,region_graph,sd,input_layer",
    list(
        itertools.product(
            [9, 12],
            [1, 4],
            [1, 3],
            ["rnd-bt", "qt"],
            [False, True],
            ["categorical", "embedding"],
        )
    ),
)
def test_discrete_sos_pc(
    num_variables, num_squares, num_units, region_graph, sd, input_layer
):
    input_layer_kwargs = (
        {"num_categories": 2} if input_layer == "categorical" else {"num_states": 2}
    )
    if region_graph == "qt":
        if num_variables == 9:
            image_shape = (1, 3, 3)
        else:  # num_variables == 12
            image_shape = (1, 4, 3)
    else:
        image_shape = None
    model = SOS(
        num_variables,
        image_shape=image_shape,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer=input_layer,
        input_layer_kwargs=input_layer_kwargs,
        num_squares=num_squares,
        region_graph=region_graph,
        structured_decomposable=sd,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "num_variables,num_squares,num_units,region_graph,sd,input_layer",
    list(
        itertools.product(
            [9, 12],
            [1, 4],
            [1, 3],
            ["rnd-bt", "qt"],
            [False, True],
            ["categorical", "embedding"],
        )
    ),
)
def test_discrete_complex_sos_pc(
    num_variables, num_squares, num_units, region_graph, sd, input_layer
):
    input_layer_kwargs = (
        {"num_categories": 2} if input_layer == "categorical" else {"num_states": 2}
    )
    if region_graph == "qt":
        if num_variables == 9:
            image_shape = (1, 3, 3)
        else:  # num_variables == 12
            image_shape = (1, 4, 3)
    else:
        image_shape = None
    model = SOS(
        num_variables,
        image_shape=image_shape,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer=input_layer,
        input_layer_kwargs=input_layer_kwargs,
        num_squares=num_squares,
        region_graph=region_graph,
        structured_decomposable=sd,
        complex=True,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "num_variables,num_units,region_graph,sd,input_layer",
    list(
        itertools.product(
            [9, 12],
            [1, 3],
            ["rnd-bt", "qt"],
            [False, True],
            ["categorical", "embedding"],
        )
    ),
)
def test_discrete_exp_sos_pc(num_variables, num_units, region_graph, sd, input_layer):
    input_layer_kwargs = (
        {"num_categories": 2} if input_layer == "categorical" else {"num_states": 2}
    )
    if region_graph == "qt":
        if num_variables == 9:
            image_shape = (1, 3, 3)
        else:  # num_variables == 12
            image_shape = (1, 4, 3)
    else:
        image_shape = None
    model = ExpSOS(
        num_variables,
        image_shape=image_shape,
        num_input_units=num_units,
        num_sum_units=num_units,
        mono_num_input_units=2,
        mono_num_sum_units=2,
        input_layer=input_layer,
        input_layer_kwargs=input_layer_kwargs,
        region_graph=region_graph,
        structured_decomposable=sd,
        complex=True,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_components,num_units,region_graph",
    list(itertools.product([1], [2], ["rnd-bt"])),
)
def test_continuous_monotonic_pc(num_components, num_units, region_graph):
    num_variables = 2
    model = MPC(
        num_variables,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer="gaussian",
        num_components=num_components,
        region_graph=region_graph,
    )
    check_pdf(model)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_squares,num_units,region_graph",
    list(itertools.product([1], [2], ["rnd-bt"])),
)
def test_continuous_sos_pc(num_squares, num_units, region_graph):
    num_variables = 2
    model = SOS(
        num_variables,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer="gaussian",
        num_squares=num_squares,
        region_graph=region_graph,
    )
    check_pdf(model)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_squares,num_units,region_graph",
    list(itertools.product([1], [2], ["rnd-bt"])),
)
def test_continuous_exp_sos_pc(num_squares, num_units, region_graph):
    num_variables = 2
    model = ExpSOS(
        num_variables,
        num_input_units=num_units,
        num_sum_units=num_units,
        mono_num_input_units=2,
        mono_num_sum_units=2,
        input_layer="gaussian",
        region_graph=region_graph,
    )
    check_pdf(model)
