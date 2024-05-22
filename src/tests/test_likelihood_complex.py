import itertools

import numpy as np
import pytest
import torch

from pcs.layers.mixture import BornMixtureLayer
from pcs.layers.candecomp import BornCPLayer
from pcs.layers.input import BornBinaryEmbeddings, \
    BornNormalDistribution, BornMultivariateNormalDistribution, \
    BornBSplines, BornBinomial, BornEmbeddings
from pcs.models import BornPC
from region_graph import RegionGraph, RegionNode
from region_graph.linear_tree import LinearTree
from region_graph.quad_tree import QuadTree
from region_graph.random_binary_tree import RandomBinaryTree

from tests.test_utils import generate_all_binary_samples, generate_all_ternary_samples
from tests.test_likelihood import check_evi_ll, check_mar_ll_pf, check_mar_ll_one, check_pdf


@pytest.mark.parametrize("compute_layer,num_variables,num_replicas,depth,num_components,input_mixture,exp_reparam",
                         list(itertools.product(
                             [BornCPLayer],
                             [8, 13], [1, 4], [-1, 1, 3], [1, 3], [False, True], [False, True]
                         )))
def test_complex_born_pc_random(compute_layer, num_variables, num_replicas, depth, num_components, input_mixture, exp_reparam):
    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas, depth=depth)
    init_method = 'log-normal' if exp_reparam else 'uniform'
    compute_layer_kwargs = input_layer_kwargs = {
        'exp_reparam': exp_reparam,
        'init_method': init_method,
        'complex': True
    }
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_layer_kwargs=input_layer_kwargs, compute_layer_kwargs=compute_layer_kwargs,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [BornCPLayer],
                             [(1, 3, 3)], [1, 3], [False, True]
                         )))
def test_complex_born_pc_pseudo_small_image(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(np.prod(image_shape).item()))
    check_evi_ll(model, data)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture,l2norm_reparam",
                         list(itertools.product(
                             [BornCPLayer],
                             [(1, 7, 7), (3, 28, 28)], [1, 3], [False, True], [False, True]
                         )))
def test_complex_born_pc_pseudo_large_image(compute_layer, image_shape, num_components, input_mixture, l2norm_reparam):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg, input_layer_cls=BornEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components,
        input_layer_kwargs={'num_states': 768, 'l2norm_reparam': l2norm_reparam})
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [BornCPLayer],
                             [(1, 7, 7)], [1, 3], [False, True]
                         )))
def test_complex_born_pc_image_dequantize(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg, input_layer_cls=BornNormalDistribution, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components, dequantize=True)
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    data = (data + torch.rand(*data.shape)) / 2.0
    logit_data, ldj = model._logit(data)
    unlogit_data, ildj = model._unlogit(logit_data)
    assert torch.allclose(data, unlogit_data)
    assert torch.allclose(ldj + ildj, torch.zeros(()))
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize("compute_layer,num_variables,num_components,input_mixture,num_replicas",
                         list(itertools.product(
                             [BornCPLayer],
                             [8, 13], [1, 3], [False, True], [1, 4]

                         )))
def test_complex_born_pc_linear_rg(compute_layer, num_variables, num_components, input_mixture, num_replicas):
    rg = LinearTree(num_variables, num_repetitions=num_replicas, randomize=True)
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize("compute_layer,num_variables,depth,num_components",
                         list(itertools.product(
                             [BornCPLayer],
                             [4, 7], [1, 2], [1, 3]
                         )))
def test_complex_born_binomial_pc(compute_layer, num_variables, depth, num_components):
    rg = RandomBinaryTree(num_variables, num_repetitions=1, depth=depth)
    model = BornPC(
        rg, input_layer_cls=BornBinomial, compute_layer_cls=compute_layer,
        num_components=num_components, input_layer_kwargs={'num_states': 3})
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 3]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables, arity=3)


def test_complex_normal_born_pc():
    rg = RandomBinaryTree(2, num_repetitions=1, depth=1)
    model = BornPC(
        rg, input_layer_cls=BornNormalDistribution,
        out_mixture_layer_cls=BornMixtureLayer,
        num_components=3)
    model.eval()
    check_pdf(model)


def test_complex_multivariate_normal_born_pc():
    rg = RegionGraph()
    rg.add_node(RegionNode([0, 1]))
    model = BornPC(
        rg, input_layer_cls=BornMultivariateNormalDistribution, out_mixture_layer_cls=BornMixtureLayer,
        num_components=3)
    model.eval()
    check_pdf(model)


@pytest.mark.parametrize("compute_layer,num_components,exp_reparam",
                         list(itertools.product([BornCPLayer], [2], [False, True])))
def test_complex_spline_born_pc(compute_layer, num_components, exp_reparam):
    rg = RandomBinaryTree(2, num_repetitions=1, depth=1)
    init_method = 'log-normal' if exp_reparam else 'uniform'
    model = BornPC(
        rg, input_layer_cls=BornBSplines, compute_layer_cls=compute_layer,
        num_components=num_components,
        input_layer_kwargs={'order': 2, 'num_knots': 6, 'interval': (0.0, 1.0), 'init_method': init_method},
        compute_layer_kwargs={'init_method': init_method, 'exp_reparam': exp_reparam}
    )
    model.eval()
    check_pdf(model, interval=(0.0, 1.0))
