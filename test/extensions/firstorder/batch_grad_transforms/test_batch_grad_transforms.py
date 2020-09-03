"""Tests for extension `BatchGradTransforms` (individual gradient transformations)."""

from test.automated_test import check_sizes_and_values
from test.extensions.firstorder.batch_grad_transforms.batch_grad_transforms_settings import (
    BATCHGRADTRANSFORMS_SETTINGS,
)
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import make_test_problems

import pytest
import torch

PROBLEMS = make_test_problems(BATCHGRADTRANSFORMS_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_batch_grad_transforms(problem):
    """Compare batch grad transformations with other first-order extensions.

    The following BackPACK extensions can be obtained via transformations:

    - Individual gradients
    - ℓ₂ norms
    - Second moments
    - Pairwise dot products

    All of the above are checked for correctness.

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    transforms = {
        "batch_grad": lambda g: g,
        "batch_l2": lambda g: (g ** 2).flatten(start_dim=1).sum(1),
        "sum_grad_squared": lambda g: (g ** 2).sum(0),
        "batch_dot": lambda g: torch.einsum(
            "ni,mi->nm", g.flatten(start_dim=1), g.flatten(start_dim=1)
        ),
    }
    backpack_res = BackpackExtensions(problem).batch_grad_transforms(transforms)

    autograd_res = {
        "batch_grad": AutogradExtensions(problem).batch_grad(),
        "batch_l2": AutogradExtensions(problem).batch_l2_grad(),
        "sum_grad_squared": AutogradExtensions(problem).sgs(),
        "batch_dot": AutogradExtensions(problem).batch_dot_grad(),
    }

    for key in transforms.keys():
        backpack_list = [result[key] for result in backpack_res]
        autograd_list = autograd_res[key]
        check_sizes_and_values(autograd_list, backpack_list)

    problem.tear_down()
