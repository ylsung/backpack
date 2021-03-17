"""Tests for `backpack.core.extensions.firstorder.batch_grad.BatchGrad`

Test individual gradients for the following layers:
- batch gradients of linear layers
- batch gradients of convolutional layers

"""
from test.automated_test import check_sizes_and_values
from test.extensions.firstorder.batch_grad.batchgrad_settings import BATCHGRAD_SETTINGS
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import make_test_problems

import pytest

PROBLEMS = make_test_problems(BATCHGRAD_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

SUBSAMPLINGS = [None, [1]]
SUBSAMPLINGS_IDS = [f"subsampling={subsampling}" for subsampling in SUBSAMPLINGS]


@pytest.mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLINGS_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_batch_grad(problem, subsampling):
    """Test individual gradients

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
        subsampling ([int]): Sample indices for which the transposed Jacobian
            is applied. If ``None``, apply to all samples in mini-batch.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).batch_grad(subsampling=subsampling)
    autograd_res = AutogradExtensions(problem).batch_grad(subsampling=subsampling)

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
