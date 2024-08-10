import numpy as np
import torch

from models import MPC, PC, SOS, ExpSOS
from sampling import inverse_transform_sample
from tests.test_utils import generate_all_binary_samples


def check_sampling(model: PC, num_samples: int):
    assert model.num_channels == 1
    samples = inverse_transform_sample(model, vdomain=2, num_samples=num_samples)
    num_variables = model.num_variables
    assert samples.shape == (num_samples, 1, num_variables)
    assert torch.all(torch.isin(samples, torch.tensor([0, 1])))

    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    prob_zero_input = torch.exp(model.log_likelihood(data.unsqueeze(dim=1)))[0].item()
    num_zero_samples = torch.count_nonzero(
        torch.all(
            samples[:, 0, :] == torch.zeros(num_variables, dtype=samples.dtype), dim=1
        )
    ).item()
    assert np.isclose(prob_zero_input, num_zero_samples / num_samples, atol=3e-3)


def test_sampling_monotonic_pc():
    num_variables = 4
    model = MPC(
        num_variables,
        num_input_units=2,
        num_sum_units=2,
        input_layer="categorical",
        input_layer_kwargs={"num_categories": 2},
        num_components=3,
        region_graph="rnd-bt",
        structured_decomposable=True,
    )
    num_samples = 10000
    check_sampling(model, num_samples)


def test_sampling_sos_pc():
    num_variables = 4
    model = SOS(
        num_variables,
        num_input_units=2,
        num_sum_units=2,
        input_layer="categorical",
        input_layer_kwargs={"num_categories": 2},
        num_squares=2,
        region_graph="rnd-bt",
        structured_decomposable=True,
    )
    num_samples = 10000
    check_sampling(model, num_samples)


def test_sampling_exp_sos_pc():
    num_variables = 4
    model = ExpSOS(
        num_variables,
        num_input_units=2,
        num_sum_units=2,
        input_layer="categorical",
        input_layer_kwargs={"num_categories": 2},
        region_graph="rnd-bt",
        structured_decomposable=True,
    )
    num_samples = 10000
    check_sampling(model, num_samples)
