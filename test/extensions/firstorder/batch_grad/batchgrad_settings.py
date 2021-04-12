"""Test cases for ``backpack.extensions.BatchGrad``.

The cases are taken from ``test.extensions.firstorder.firstorder_settings``.
Additional local cases can be defined by appending them to ``LOCAL_SETTINGS``.
"""

from test.core.derivatives.utils import regression_targets
from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS

import torch

BATCHGRAD_SETTINGS = []

SHARED_SETTINGS = FIRSTORDER_SETTINGS
LOCAL_SETTINGS = [
    # nn.Linear with one additional dimension
    {
        "input_fn": lambda: torch.rand(3, 4, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 3), torch.nn.Linear(3, 2)
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4, 2)),
        "id_prefix": "one-additional",
    },
    # nn.Linear with two additional dimensions
    {
        "input_fn": lambda: torch.rand(3, 4, 2, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 3), torch.nn.Linear(3, 2)
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="mean"),
        "target_fn": lambda: regression_targets((3, 4, 2, 2)),
        "id_prefix": "two-additional",
    },
    # nn.Linear with three additional dimensions, sum reduction
    {
        "input_fn": lambda: torch.rand(3, 4, 2, 3, 5),
        "module_fn": lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 3), torch.nn.Linear(3, 2)
        ),
        "loss_function_fn": lambda: torch.nn.MSELoss(reduction="sum"),
        "target_fn": lambda: regression_targets((3, 4, 2, 3, 2)),
        "id_prefix": "three-additional",
    },
]

BATCHGRAD_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
