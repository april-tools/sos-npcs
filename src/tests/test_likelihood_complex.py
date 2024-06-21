import itertools

import numpy as np
import pytest
import torch

from pcs.layers.candecomp import BornCPLayer
from pcs.layers.input import (BornBinaryEmbeddings, BornBinomial, BornBSplines,
                              BornEmbeddings, BornNormalDistribution)
from pcs.layers.mixture import BornMixtureLayer
from pcs.models import BornPC
from region_graph import RegionGraph, RegionNode
from region_graph.linear_tree import LinearTree
from region_graph.quad_tree import QuadTree
from region_graph.random_binary_tree import RandomBinaryTree
from tests.test_likelihood import (check_evi_ll, check_mar_ll_one,
                                   check_mar_ll_pf, check_pdf)
from tests.test_utils import (generate_all_binary_samples,
                              generate_all_ternary_samples)


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
def test_complex_born_pc_random(
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
        "complex": True,
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
    "compute_layer,image_shape,num_units",
    list(itertools.product([BornCPLayer], [(1, 3, 3)], [1, 3])),
)
def test_complex_born_pc_pseudo_small_image(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    compute_layer_kwargs = input_layer_kwargs = {"complex": True}
    model = BornPC(
        rg,
        input_layer_cls=BornBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs=input_layer_kwargs,
        compute_layer_kwargs=compute_layer_kwargs,
    )
    data = torch.LongTensor(generate_all_binary_samples(np.prod(image_shape).item()))
    check_evi_ll(model, data)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units,l2norm_reparam",
    list(
        itertools.product(
            [BornCPLayer], [(1, 7, 7), (3, 28, 28)], [1, 3], [False, True]
        )
    ),
)
def test_complex_born_pc_pseudo_large_image(
    compute_layer, image_shape, num_units, l2norm_reparam
):
    rg = QuadTree(image_shape, struct_decomp=True)
    compute_layer_kwargs = {"complex": True}
    input_layer_kwargs = {
        "num_states": 768,
        "complex": True,
        "l2norm_reparam": l2norm_reparam,
    }
    model = BornPC(
        rg,
        input_layer_cls=BornEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs=input_layer_kwargs,
        compute_layer_kwargs=compute_layer_kwargs,
    )
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize(
    "compute_layer,image_shape,num_units",
    list(itertools.product([BornCPLayer], [(1, 7, 7)], [1, 3])),
)
def test_complex_born_pc_image_dequantize(compute_layer, image_shape, num_units):
    rg = QuadTree(image_shape, struct_decomp=True)
    compute_layer_kwargs = {"complex": True}
    model = BornPC(
        rg,
        input_layer_cls=BornNormalDistribution,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        dequantize=True,
        compute_layer_kwargs=compute_layer_kwargs,
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
    list(itertools.product([BornCPLayer], [8, 13], [1, 3], [1, 4])),
)
def test_complex_born_pc_linear_rg(
    compute_layer, num_variables, num_units, num_replicas
):
    rg = LinearTree(num_variables, num_repetitions=num_replicas, randomize=True)
    compute_layer_kwargs = input_layer_kwargs = {"complex": True}
    model = BornPC(
        rg,
        input_layer_cls=BornBinaryEmbeddings,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        compute_layer_kwargs=compute_layer_kwargs,
        input_layer_kwargs=input_layer_kwargs,
    )
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize(
    "compute_layer,num_variables,num_units",
    list(itertools.product([BornCPLayer], [4, 7], [1, 3])),
)
def test_complex_born_binomial_pc(compute_layer, num_variables, num_units):
    rg = RandomBinaryTree(num_variables, num_repetitions=1)
    model = BornPC(
        rg,
        input_layer_cls=BornBinomial,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        compute_layer_kwargs={"complex": True},
        input_layer_kwargs={"num_states": 3},
    )
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 3]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables, arity=3)


def test_complex_normal_born_pc():
    rg = RandomBinaryTree(2, num_repetitions=1)
    model = BornPC(
        rg,
        input_layer_cls=BornNormalDistribution,
        out_mixture_layer_cls=BornMixtureLayer,
        num_units=3,
        compute_layer_kwargs={"complex": True},
    )
    model.eval()
    check_pdf(model)


@pytest.mark.parametrize(
    "compute_layer,num_units,exp_reparam",
    list(itertools.product([BornCPLayer], [2], [False, True])),
)
def test_complex_spline_born_pc(compute_layer, num_units, exp_reparam):
    rg = RandomBinaryTree(2, num_repetitions=1)
    init_method = "log-normal" if exp_reparam else "normal"
    model = BornPC(
        rg,
        input_layer_cls=BornBSplines,
        compute_layer_cls=compute_layer,
        num_units=num_units,
        input_layer_kwargs={
            "order": 2,
            "num_knots": 6,
            "interval": (0.0, 1.0),
            "init_method": init_method,
            "complex": True,
        },
        compute_layer_kwargs={
            "init_method": init_method,
            "complex": True,
            "exp_reparam": exp_reparam,
        },
    )
    model.eval()
    check_pdf(model, interval=(0.0, 1.0))
