import itertools
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from scipy import integrate

from pcs.hmm import BornHMM, MonotonicHMM
from pcs.layers.candecomp import BornCPLayer, MonotonicCPLayer
from pcs.layers.input import (
    BornBinaryEmbeddings,
    BornBinomial,
    BornBSplines,
    BornEmbeddings,
    BornNormalDistribution,
    MonotonicBinaryEmbeddings,
    MonotonicBinomial,
    MonotonicBSplines,
    MonotonicEmbeddings,
    NormalDistribution,
)
from pcs.layers.mixture import BornMixtureLayer
from pcs.models import PC, BornPC, MonotonicPC
from region_graph.linear_tree import LinearTree
from region_graph.quad_tree import QuadTree
from region_graph.random_binary_tree import RandomBinaryTree
from tests.test_utils import generate_all_binary_samples, generate_all_ternary_samples


def check_normalized_log_scores(model: PC, x: torch.Tensor) -> torch.Tensor:
    scores = model.log_score(x)
    assert scores.shape == (len(x), 1)
    assert torch.all(torch.isfinite(scores))
    assert torch.allclose(
        torch.logsumexp(scores, dim=0).exp(), torch.tensor(1.0), atol=1e-15
    )
    return scores


def check_evi_ll(model: PC, x: torch.Tensor) -> torch.Tensor:
    lls = model.log_prob(x)
    assert lls.shape == (len(x), 1)
    assert torch.all(torch.isfinite(lls))
    assert torch.allclose(
        torch.logsumexp(lls, dim=0).exp(), torch.tensor(1.0), atol=1e-15
    )
    return lls


def check_mar_ll_pf(model: PC, x: torch.Tensor):
    mar_mask = torch.ones_like(x, dtype=torch.bool)
    lls = model.log_marginal_score(x, mar_mask)
    log_z = model.log_pf()
    assert torch.allclose(lls, log_z, atol=1e-15)


def check_mar_ll_one(
    model: PC, x: torch.Tensor, num_mar_variables: int = 1, arity: int = 2
) -> torch.Tensor:
    assert x.shape[1] > num_mar_variables
    num_mar_samples = arity**num_mar_variables
    z = x[:num_mar_samples]
    mar_mask = torch.zeros_like(z, dtype=torch.bool)
    mar_mask[:, : z.shape[1] - num_mar_variables] = True
    lls = model.log_marginal_prob(z, mar_mask)
    assert torch.allclose(
        torch.logsumexp(lls, dim=0).exp(), torch.tensor(1.0), atol=1e-15
    )
    return lls


def check_pdf(model, interval: Optional[Tuple[float, float]] = None):
    pdf = lambda y, x: torch.exp(model.log_prob(torch.Tensor([[x, y]])))
    if interval is None:
        a, b = -64.0, 64.0
    else:
        a, b = interval
    ig, err = integrate.dblquad(pdf, a, b, a, b)
    assert np.isclose(ig, 1.0, atol=1e-15)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_replicas,num_units",
    list(itertools.product([MonotonicCPLayer], [8, 13], [1, 4], [1, 3])),
)
def test_monotonic_pc_random(compute_layer, num_variables, num_replicas, num_units):
    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas)
    model = MonotonicPC(
        rg,
        input_layer_cls=MonotonicBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_replicas,num_units,exp_reparam",
    list(
        itertools.product(
            [BornCPLayer],
            [8, 13],
            [1, 4],
            [1, 3],
            [False, True],
        )
    ),
)
def test_born_pc_random(
    compute_layer,
    num_variables,
    num_replicas,
    num_units,
    exp_reparam,
):
    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas)
    init_method = "log-normal" if exp_reparam else "normal"
    compute_layer_kwargs = input_layer_kwargs = {
        "exp_reparam": exp_reparam,
        "init_method": init_method,
    }
    model = BornPC(
        rg,
        input_layer_cls=BornBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        input_layer_kwargs=input_layer_kwargs,
        compute_layer_kwargs=compute_layer_kwargs,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_replicas,num_units",
    list(itertools.product([BornCPLayer], [3, 8], [1, 4], [1, 2])),
)
def test_born_pc_linear_stiefel(compute_layer, num_variables, num_replicas, num_units):
    rg = LinearTree(num_variables, num_repetitions=num_replicas)
    compute_layer_kwargs = {"init_method": "stiefel"}
    input_layer_kwargs = {"init_method": "stiefel", "num_states": 3}
    model = BornPC(
        rg,
        input_layer_cls=BornEmbeddings,
        compute_layer_cls=compute_layer,
        input_layer_kwargs=input_layer_kwargs,
        compute_layer_kwargs=compute_layer_kwargs,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_normalized_log_scores(model, data)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units",
    list(itertools.product([MonotonicCPLayer], [(1, 3, 3)], [1, 3])),
)
def test_monotonic_pc_pseudo_small_image(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = MonotonicPC(
        rg,
        input_layer_cls=MonotonicBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_binary_samples(np.prod(image_shape).item()))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units",
    list(itertools.product([BornCPLayer], [(1, 3, 3)], [1, 3])),
)
def test_born_pc_pseudo_small_image(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg,
        input_layer_cls=BornBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_binary_samples(np.prod(image_shape).item()))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units",
    list(itertools.product([MonotonicCPLayer], [(1, 7, 7), (3, 28, 28)], [1, 3])),
)
def test_monotonic_pc_pseudo_large_image(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = MonotonicPC(
        rg,
        input_layer_cls=MonotonicEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs={"num_states": 768},
    )
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units,l2norm_reparam",
    list(
        itertools.product(
            [BornCPLayer],
            [(1, 7, 7), (3, 28, 28)],
            [1, 3],
            [False, True],
        )
    ),
)
def test_born_pc_pseudo_large_image(
    compute_layer, image_shape, num_units, l2norm_reparam
):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg,
        input_layer_cls=BornEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs={"num_states": 768, "l2norm_reparam": l2norm_reparam},
    )
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units",
    list(itertools.product([MonotonicCPLayer], [(1, 7, 7)], [1, 3])),
)
def test_monotonic_pc_image_dequantize(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = MonotonicPC(
        rg,
        input_layer_cls=NormalDistribution,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        dequantize=True,
    )
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    data = (data + torch.rand(*data.shape)) / 2.0
    logit_data, ldj = model._logit(data)
    unlogit_data, ildj = model._unlogit(logit_data)
    assert torch.allclose(data, unlogit_data)
    assert torch.allclose(ldj + ildj, torch.zeros(()))
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units",
    list(itertools.product([BornCPLayer], [(1, 7, 7)], [1, 3])),
)
def test_born_pc_image_dequantize(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg,
        input_layer_cls=BornNormalDistribution,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        dequantize=True,
    )
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    data = (data + torch.rand(*data.shape)) / 2.0
    logit_data, ldj = model._logit(data)
    unlogit_data, ildj = model._unlogit(logit_data)
    assert torch.allclose(data, unlogit_data)
    assert torch.allclose(ldj + ildj, torch.zeros(()))
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_units,num_replicas",
    list(itertools.product([MonotonicCPLayer], [8, 13], [1, 3], [1, 4])),
)
def test_monotonic_pc_linear_rg(compute_layer, num_variables, num_units, num_replicas):
    rg = LinearTree(num_variables, num_repetitions=num_replicas, randomize=True)
    model = MonotonicPC(
        rg,
        input_layer_cls=MonotonicBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_units,num_replicas",
    list(itertools.product([BornCPLayer], [8, 13], [1, 3], [1, 4])),
)
def test_born_pc_linear_rg(compute_layer, num_variables, num_units, num_replicas):
    rg = LinearTree(num_variables, num_repetitions=num_replicas, randomize=True)
    model = BornPC(
        rg,
        input_layer_cls=BornBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_units",
    list(itertools.product([MonotonicCPLayer], [4, 7], [1, 3])),
)
def test_monotonic_binomial_pc(compute_layer, num_variables, num_units):
    rg = RandomBinaryTree(num_variables, num_repetitions=1)
    model = MonotonicPC(
        rg,
        input_layer_cls=MonotonicBinomial,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs={"num_states": 3},
    )
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 3]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables, arity=3)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_units",
    list(itertools.product([BornCPLayer], [4, 7], [1, 3])),
)
def test_born_binomial_pc(compute_layer, num_variables, num_units):
    rg = RandomBinaryTree(num_variables, num_repetitions=1)
    model = BornPC(
        rg,
        input_layer_cls=BornBinomial,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs={"num_states": 3},
    )
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 3]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables, arity=3)


@pytest.mark.parametrize(
    "compute_layer,num_units", list(itertools.product([MonotonicCPLayer], [3]))
)
def test_normal_monotonic_pc(compute_layer, num_units):
    rg = RandomBinaryTree(2, num_repetitions=1)
    model = MonotonicPC(
        rg,
        input_layer_cls=NormalDistribution,
        compute_layer_cls=compute_layer,
        num_units=num_units,
    )
    model.eval()
    check_pdf(model)


def test_normal_born_pc():
    rg = RandomBinaryTree(2, num_repetitions=1)
    model = BornPC(
        rg,
        input_layer_cls=BornNormalDistribution,
        out_mixture_layer_cls=BornMixtureLayer,
        num_units=3,
    )
    model.eval()
    check_pdf(model)


@pytest.mark.parametrize(
    "compute_layer,num_units", list(itertools.product([MonotonicCPLayer], [2]))
)
def test_spline_monotonic_pc(compute_layer, num_units):
    rg = RandomBinaryTree(2, num_repetitions=1)
    model = MonotonicPC(
        rg,
        input_layer_cls=MonotonicBSplines,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        num_input_units=6,
        input_layer_kwargs={"order": 2, "interval": (0.0, 1.0)},
    )
    model.eval()
    check_pdf(model, interval=(0.0, 1.0))


@pytest.mark.parametrize(
    "compute_layer,num_units,exp_reparam",
    list(itertools.product([BornCPLayer], [2], [False, True])),
)
def test_spline_born_pc(compute_layer, num_units, exp_reparam):
    rg = RandomBinaryTree(2, num_repetitions=1)
    init_method = "log-normal" if exp_reparam else "normal"
    model = BornPC(
        rg,
        input_layer_cls=BornBSplines,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        num_input_units=6,
        input_layer_kwargs={"order": 2, "interval": (0.0, 1.0)},
        compute_layer_kwargs={"init_method": init_method, "exp_reparam": exp_reparam},
    )
    model.eval()
    check_pdf(model, interval=(0.0, 1.0))


@pytest.mark.parametrize(
    "seq_length,hidden_size", list(itertools.product([2, 7], [1, 13]))
)
def test_monotonic_hmm(seq_length, hidden_size):
    model = MonotonicHMM(vocab_size=3, seq_length=seq_length, hidden_size=hidden_size)
    data = torch.LongTensor(generate_all_ternary_samples(seq_length))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "seq_length,hidden_size,l2norm_reparam",
    list(itertools.product([2, 7], [1, 13], [False, True])),
)
def test_born_hmm(seq_length, hidden_size, l2norm_reparam):
    model = BornHMM(
        vocab_size=3,
        seq_length=seq_length,
        hidden_size=hidden_size,
        l2norm_reparam=l2norm_reparam,
    )
    data = torch.LongTensor(generate_all_ternary_samples(seq_length))
    check_evi_ll(model, data)
