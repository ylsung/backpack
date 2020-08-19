"""Tests for extension `BatchDotGrad` (individual gradients dot products)."""

from test.automated_test import check_sizes_and_values
from test.extensions.firstorder.batch_dot_grad.batch_dot_grad_settings import (
    BATCHDOTGRAD_SETTINGS,
)
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import make_test_problems

import pytest

PROBLEMS = make_test_problems(BATCHDOTGRAD_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_batch_dot_grad(problem):
    """Test individual gradient pairwise dot product.

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    # TODO: Compute and compare pairwise dot products
    print("This is a dummy")
    pass
